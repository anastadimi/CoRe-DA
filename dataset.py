import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import normalize_label


class MixingDataset(Dataset):
    """
    Training-only dataset returning: (no mixing behaviour during training in the original paper)
        - S (source), mixed with backgrounds from T
        - ex (exemplar), mixed with backgrounds from T
        - T (target), mixed with backgrounds from S

    Background types (mutually exclusive per branch):
        - video-level background (precomputed image)
        - clip-level background (median of snippet_size frames)

    Controlled via 6 flags:
        use_video_BG_in_S, use_clip_BG_in_S
        use_video_BG_in_ex, use_clip_BG_in_ex
        use_video_BG_in_T, use_clip_BG_in_T

    """

    def __init__(
        self,
        source_db_json_file,
        target_db_json_file,
        label_column: str,
        data_root_dir: str,
        num_segments: int,
        snippet_size: int,
        norm_stats,
        ordered_sampling: bool = True,
        # label handling
        norm_labels: bool = True,
        # Background flags (mutually exclusive per branch)
        use_video_BG_in_S: bool = False,
        use_clip_BG_in_S: bool = False,
        use_video_BG_in_ex: bool = False,
        use_clip_BG_in_ex: bool = False,
        use_video_BG_in_T: bool = False,
        use_clip_BG_in_T: bool = False,
        # Mixing
        gamma_alpha: float = 0.5,
        # Debug saving
        save_mixed_debug: bool = False,
        debug_output_root: str = "debug_mixed",
    ):
        super().__init__()

        # DBs
        self.source_db = source_db_json_file
        self.target_db = target_db_json_file
        self.label_column = label_column
        self.data_root_dir = data_root_dir

        # Clip sampling
        self.num_segments = num_segments
        self.snippet_size = snippet_size
        self.ordered_sampling = ordered_sampling

        # Label normalization
        self.norm_labels = norm_labels

        # BG flags
        self.use_video_BG_in_S = use_video_BG_in_S
        self.use_clip_BG_in_S = use_clip_BG_in_S

        self.use_video_BG_in_ex = use_video_BG_in_ex
        self.use_clip_BG_in_ex = use_clip_BG_in_ex

        self.use_video_BG_in_T = use_video_BG_in_T
        self.use_clip_BG_in_T = use_clip_BG_in_T

        self._validate_bg_flags()

        # Mixing params
        self.gamma_alpha = gamma_alpha

        # Debug saving
        self.save_mixed_debug = save_mixed_debug
        self.debug_output_root = debug_output_root

        # Transforms
        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=norm_stats["mean"], std=norm_stats["std"])]
        )

    def _validate_bg_flags(self):
        def check(name, v, c):
            if v and c:
                raise ValueError(
                    f"Branch '{name}' cannot use BOTH video and clip backgrounds. "
                    f"Enable only one."
                )

        check("S", self.use_video_BG_in_S, self.use_clip_BG_in_S)
        check("ex", self.use_video_BG_in_ex, self.use_clip_BG_in_ex)
        check("T", self.use_video_BG_in_T, self.use_clip_BG_in_T)

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.source_db)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Returns:
            {
                "S":  (tensor_S, label_S),
                "ex": (tensor_ex, label_ex),
                "T":   tensor_T
            }
        """

        # ==============================================================
        #                        LOAD SOURCE
        # ==============================================================
        S_item = self.source_db[idx]
        S_frames_all = S_item["frames_path"]
        S_bg_video = S_item["bg_frame_path"]
        label_S = torch.tensor(S_item[self.label_column], dtype=torch.float)
        label_S = (
            normalize_label(label_S, label_column=self.label_column, denormalize=False)
            if self.norm_labels
            else label_S
        )

        # ==============================================================
        #                        LOAD EXEMPLAR (random S)
        # ==============================================================
        ex_idx = np.random.randint(0, len(self.source_db))
        if ex_idx == idx:
            ex_idx = (ex_idx + 1) % len(self.source_db)
        ex_item = self.source_db[ex_idx]
        ex_frames_all = ex_item["frames_path"]
        ex_bg_video = ex_item["bg_frame_path"]
        label_ex = torch.tensor(ex_item[self.label_column], dtype=torch.float)
        label_ex = (
            normalize_label(label_ex, label_column=self.label_column, denormalize=False)
            if self.norm_labels
            else label_ex
        )

        # ==============================================================
        #                        LOAD TARGET
        # ==============================================================
        T_idx = np.random.randint(0, len(self.target_db))
        T_item = self.target_db[T_idx]
        T_frames_all = T_item["frames_path"]
        T_bg_video = T_item["bg_frame_path"]
        label_T = torch.tensor(T_item[self.label_column], dtype=torch.float)
        label_T = (
            normalize_label(label_T, label_column=self.label_column, denormalize=False)
            if self.norm_labels
            else label_T
        )

        # ==============================================================
        #                        SAMPLE CLIPS
        # ==============================================================
        S_clips = self._sample_clips(S_frames_all)
        ex_clips = self._sample_clips(ex_frames_all)
        T_clips = self._sample_clips(T_frames_all)
        # returns list of list of frame paths [[clip1_frame1_path,...,clip1_frameK_path],...,[...]]

        # ==============================================================
        #                        VIDEO-LEVEL BGS
        # ==============================================================
        bg_video_T = self._read_bgr(T_bg_video)
        bg_video_S = self._read_bgr(S_bg_video)

        # ==============================================================
        #                 CONDITIONAL CLIP-LEVEL BGS
        # ==============================================================
        bg_clip_T = None
        if self.use_clip_BG_in_S or self.use_clip_BG_in_ex:
            bg_clip_T = self._compute_clip_backgrounds(T_clips)

        bg_clip_S = None
        if self.use_clip_BG_in_T:
            bg_clip_S = self._compute_clip_backgrounds(S_clips)

        # ==============================================================
        #                        SELECT BACKGROUNDS
        # ==============================================================
        # S mixes with T BG
        bg_for_S = self._select_backgrounds(
            from_video=bg_video_T,
            from_clips=bg_clip_T,
            use_video_flag=self.use_video_BG_in_S,
            use_clip_flag=self.use_clip_BG_in_S,
        )

        # ex mixes with T BG
        bg_for_ex = self._select_backgrounds(
            from_video=bg_video_T,
            from_clips=bg_clip_T,
            use_video_flag=self.use_video_BG_in_ex,
            use_clip_flag=self.use_clip_BG_in_ex,
        )

        # T mixes with S BG
        bg_for_T = self._select_backgrounds(
            from_video=bg_video_S,
            from_clips=bg_clip_S,
            use_video_flag=self.use_video_BG_in_T,
            use_clip_flag=self.use_clip_BG_in_T,
        )

        # ==============================================================
        #                        MIX + TRANSFORM
        # ==============================================================
        tensor_S = self._clips_to_tensor(S_clips, bg_for_S)
        tensor_ex = self._clips_to_tensor(ex_clips, bg_for_ex)
        tensor_T = self._clips_to_tensor(T_clips, bg_for_T)

        # ==============================================================
        #                        OPTIONAL DEBUG SAVE
        # ==============================================================
        if self.save_mixed_debug:
            S_name = S_item.get("video_id", f"S_{idx}")
            ex_name = ex_item.get("video_id", f"ex_{ex_idx}")
            T_name = T_item.get("video_id", f"T_{T_idx}")

            self._save_mixed_clips(S_clips, bg_for_S, S_name, "S")
            self._save_mixed_clips(ex_clips, bg_for_ex, ex_name, "ex")
            self._save_mixed_clips(T_clips, bg_for_T, T_name, "T")

        # ==============================================================
        return {
            "S": (tensor_S.permute(1, 0, 2, 3), label_S),
            "ex": (tensor_ex.permute(1, 0, 2, 3), label_ex),
            "T": (tensor_T.permute(1, 0, 2, 3), label_T),
        }

    # =====================================================================
    #                    BACKGROUND SELECTOR (video XOR clip)
    # =====================================================================
    def _select_backgrounds(
        self, from_video, from_clips, use_video_flag, use_clip_flag
    ):
        if not use_video_flag and not use_clip_flag:
            return None

        if use_video_flag:
            return [[from_video] for _ in range(self.num_segments)]

        if use_clip_flag:
            return [[from_clips[i]] for i in range(self.num_segments)]

    # =====================================================================
    #                        MIXING → TENSOR
    # =====================================================================
    def _clips_to_tensor(self, clips, bg_per_clip):
        # --------------------------------------------------------------
        # SAMPLE ONE alpha FOR THE WHOLE VIDEO
        # --------------------------------------------------------------
        alpha_global = np.random.uniform(0, self.gamma_alpha)

        frames = []

        for clip_idx, clip_paths in enumerate(clips):

            BG_list = bg_per_clip[clip_idx] if bg_per_clip is not None else []

            for fp in clip_paths:
                bgr = cv2.imread(fp)
                if bgr is None:
                    raise FileNotFoundError(fp)

                H, W = bgr.shape[:2]
                mixed = bgr.astype(np.float32)

                for bg in BG_list:  # at most one
                    bg_r = cv2.resize(bg, (W, H)).astype(np.float32)

                    # ------------------------------------------------------
                    # USE THE SAME α FOR THE ENTIRE VIDEO
                    # ------------------------------------------------------
                    mixed = (1 - alpha_global) * mixed + alpha_global * bg_r

                mixed = np.clip(mixed, 0, 255).astype(np.uint8)
                img = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
                frames.append(self.transform(img))

        return torch.stack(frames, dim=0)

    # =====================================================================
    #                          CLIP SAMPLING
    # =====================================================================
    def _sample_clips(self, frames_all):
        n = len(frames_all)
        clips = []

        if self.ordered_sampling:
            seg_size = max(1, n // self.num_segments)
            for i in range(self.num_segments):
                s = i * seg_size
                e = min(s + self.snippet_size, n)
                clip = frames_all[s:e]
                if len(clip) < self.snippet_size:
                    clip += [frames_all[-1]] * (self.snippet_size - len(clip))
                clips.append([os.path.join(self.data_root_dir, p) for p in clip])
        else:
            max_start = max(0, n - self.snippet_size)
            for _ in range(self.num_segments):
                s = np.random.randint(0, max_start + 1)
                clip = frames_all[s : s + self.snippet_size]
                clips.append([os.path.join(self.data_root_dir, p) for p in clip])

        return clips

    # =====================================================================
    #                    CLIP-LEVEL BACKGROUND
    # =====================================================================
    def _compute_clip_backgrounds(self, clips):
        bgs = []
        for clip in clips:
            frames_np = []
            for fp in clip:
                rgb = Image.open(fp).convert("RGB")
                frames_np.append(np.asarray(rgb, dtype=np.uint8))

            arr = np.stack(frames_np, axis=0)
            med_r = np.median(arr[:, :, :, 0], axis=0)
            med_g = np.median(arr[:, :, :, 1], axis=0)
            med_b = np.median(arr[:, :, :, 2], axis=0)
            bg_rgb = np.stack([med_r, med_g, med_b], axis=-1).astype(np.uint8)

            bgs.append(bg_rgb[:, :, ::-1])  # RGB → BGR

        return bgs

    # =====================================================================
    #                          DEBUG SAVING
    # =====================================================================
    def _save_mixed_clips(self, clips, bg_per_clip, video_name, prefix):
        out_root = os.path.join(self.debug_output_root, video_name, prefix)
        os.makedirs(out_root, exist_ok=True)

        # --------------------------------------------------------------
        # ONE α FOR THE ENTIRE VIDEO (same logic as _clips_to_tensor)
        # --------------------------------------------------------------
        alpha_global = np.random.uniform(0, self.gamma_alpha)

        for clip_idx, clip_paths in enumerate(clips):

            clip_dir = os.path.join(out_root, f"clip_{clip_idx}")
            os.makedirs(clip_dir, exist_ok=True)

            BG_list = bg_per_clip[clip_idx] if bg_per_clip is not None else []

            for f_idx, fp in enumerate(clip_paths):
                bgr = cv2.imread(fp)
                H, W = bgr.shape[:2]
                mixed = bgr.astype(np.float32)

                for bg in BG_list:
                    bg_r = cv2.resize(bg, (W, H)).astype(np.float32)

                    # ------------------------------------------------------
                    # USE THE SAME α FOR THE ENTIRE VIDEO
                    # ------------------------------------------------------
                    mixed = (1 - alpha_global) * mixed + alpha_global * bg_r

                mixed = np.clip(mixed, 0, 255).astype(np.uint8)
                out_path = os.path.join(clip_dir, f"{f_idx:03d}.jpg")
                cv2.imwrite(out_path, mixed)

    # =====================================================================
    #                                 UTILS
    # =====================================================================

    def _read_bgr(self, path):
        img = cv2.imread(os.path.join(self.data_root_dir, path))
        return img if img is not None else None


class TargetPlusExemplars(Dataset):
    """
    Samples N examplars from the source dataset and one target video for testing
    """

    def __init__(
        self,
        source_db_json_file,
        target_db_json_file,
        label_column: str,
        data_root_dir: str,
        num_segments: int,
        snippet_size: int,
        norm_stats,
        n_exemplars,
        sampling_method,
        norm_labels: bool = True,
        gamma_alpha: float = 0.5,
        mix_test: bool = False,
        save_mixed_debug: bool = False,
        debug_output_root: str = "debug_target",
    ):
        super().__init__()

        self.source_db = source_db_json_file
        self.target_db = target_db_json_file
        self.label_column = label_column
        self.data_root_dir = data_root_dir

        self.num_segments = num_segments
        self.snippet_size = snippet_size
        self.norm_labels = norm_labels
        self.gamma_alpha = gamma_alpha
        self.mix_test = mix_test

        self.save_mixed_debug = save_mixed_debug
        self.debug_output_root = debug_output_root

        # transforms for final model input
        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=norm_stats["mean"], std=norm_stats["std"])]
        )

        # exemplar selection
        self.n_exemplars = n_exemplars
        self.closest_exemplar_cache = {}  # cache for closest-label sampling

        if sampling_method == "uniform":
            self.idx_exemplars = self._uniform_sampling()
        elif sampling_method == "distribution":
            self.idx_exemplars = self._distribution_sampling()
        elif sampling_method == "random":
            self.idx_exemplars = self._random_sampling()
        elif (
            sampling_method == "closest_label"
        ):  # oracle behavior (only used for debugging)
            self.target_to_exemplars = self._precompute_closest_label_map()
            self.idx_exemplars = None  # not used directly
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    # ================================================================
    def __len__(self):
        return len(self.target_db)

    # ================================================================
    def __getitem__(self, idx):

        # --------------------------------------------------------------
        # 1. Load target video all frames (padded)
        # --------------------------------------------------------------
        T_item = self.target_db[idx]
        label_T = torch.tensor(T_item[self.label_column], dtype=torch.float)
        if self.norm_labels:
            label_T = normalize_label(label_T, self.label_column, denormalize=False)

        T_rel = self._sample_frames(T_item["frames_path"])
        T_abs = [os.path.join(self.data_root_dir, p) for p in T_rel]

        # raw uint8 RGB used for mixing
        T_raw = [self._read_rgb_uint8(fp) for fp in T_abs]
        H, W, _ = T_raw[0].shape

        # output stacks
        target_video_stack = []
        target_label_stack = []
        exemplars_video_stack = []
        exemplars_label_stack = []

        # --------------------------------------------------------------
        # 2. Loop over exemplars
        # --------------------------------------------------------------
        if hasattr(self, "target_to_exemplars"):
            exemplar_indices = self.target_to_exemplars[idx]
        else:
            exemplar_indices = self.idx_exemplars

        for ex_idx in exemplar_indices:

            ex_item = self.source_db[ex_idx]
            # print(ex_item["video"],ex_item[self.label_column])

            # exemplar label
            label_ex = torch.tensor(ex_item[self.label_column], dtype=torch.float)
            if self.norm_labels:
                label_ex = normalize_label(
                    label_ex, self.label_column, denormalize=False
                )
            exemplars_label_stack.append(label_ex)

            # exemplar full frames
            ex_rel = self._sample_frames(ex_item["frames_path"])
            ex_abs = [os.path.join(self.data_root_dir, p) for p in ex_rel]
            ex_tensor = self._read_list_to_tensor(ex_abs)
            exemplars_video_stack.append(ex_tensor)  # shape (T,C,H,W)

            # exemplar background
            bg_path = os.path.join(self.data_root_dir, ex_item["bg_frame_path"])
            bg_bgr = cv2.imread(bg_path)
            if bg_bgr is None:
                raise FileNotFoundError(bg_path)
            bg_bgr = cv2.resize(bg_bgr, (W, H)).astype(np.float32)

            # ----------------------------------------------------------
            # 3. MIX target with exemplar background
            # ----------------------------------------------------------
            alpha = self.gamma_alpha
            mixed_frames = []

            # create debug directory
            if self.save_mixed_debug:
                dbg_dir = os.path.join(
                    self.debug_output_root, f"target_{idx}", f"exemplar_{ex_idx}"
                )
                os.makedirs(dbg_dir, exist_ok=True)

                # save BG as uint8
                cv2.imwrite(
                    os.path.join(dbg_dir, "background_used.jpg"),
                    bg_bgr.astype(np.uint8),
                )

            for t_i, rgb in enumerate(T_raw):

                frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

                if self.mix_test:
                    mixed_bgr = (1 - alpha) * frame_bgr + alpha * bg_bgr
                else:
                    mixed_bgr = frame_bgr

                mixed_bgr = np.clip(mixed_bgr, 0, 255).astype(np.uint8)

                mixed_rgb = cv2.cvtColor(mixed_bgr, cv2.COLOR_BGR2RGB)

                # ------------------------------------------------------
                # HERE!!! ✅ save the image exactly as you want it
                # ------------------------------------------------------
                if self.save_mixed_debug:
                    Image.fromarray(mixed_rgb).save(
                        os.path.join(dbg_dir, f"mixed_{t_i:03d}.jpg")
                    )

                # final transformed image
                img = Image.fromarray(mixed_rgb)
                img_t = self.transform(img)  # (C,H,W)
                mixed_frames.append(img_t)

            mixed_tensor = torch.stack(mixed_frames, dim=0)  # (T,C,H,W)
            target_video_stack.append(mixed_tensor)
            target_label_stack.append(label_T)

        # --------------------------------------------------------------
        return (target_video_stack, target_label_stack), (
            exemplars_video_stack,
            exemplars_label_stack,
        )

    # ================================================================
    # Utilities
    # ================================================================

    def _sample_frames(self, frames):
        """Pad frames to multiple of snippet_size."""
        n = len(frames)
        r = n % self.snippet_size
        if r == 0:
            return frames
        pad = self.snippet_size - r
        return frames + [frames[-1]] * pad

    def _read_rgb_uint8(self, fp):
        return np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8)

    def _read_list_to_tensor(self, paths_abs):
        frames = []
        for fp in paths_abs:
            img = Image.open(fp).convert("RGB")
            frames.append(self.transform(img))
        return torch.stack(frames, dim=0)  # (T,C,H,W)

    # ================================================================
    # Sampling strategies
    # ================================================================
    def _uniform_sampling(self):
        grs_values = np.array([d["grs"] for d in self.source_db])
        min_val, max_val = grs_values.min(), grs_values.max()
        target_points = np.linspace(min_val, max_val, self.n_exemplars)

        chosen, used = [], set()
        for t in target_points:
            idx = (np.abs(grs_values - t)).argmin()
            if idx in used:
                for alt in np.argsort(np.abs(grs_values - t)):
                    if alt not in used:
                        idx = alt
                        break
            used.add(idx)
            chosen.append(idx)
        return chosen

    def _distribution_sampling(self, bins=5):
        grs_values = np.array([d["grs"] for d in self.source_db])
        counts, edges = np.histogram(grs_values, bins=bins)
        bin_idx = np.digitize(grs_values, edges, right=True)

        chosen = []
        for _ in range(self.n_exemplars):
            b = np.random.choice(bins, p=counts / counts.sum())
            candidates = np.where(bin_idx == b + 1)[0]
            if len(candidates):
                chosen.append(np.random.choice(candidates))
        return list(set(chosen))

    def _random_sampling(self):
        n = len(self.source_db)
        return np.random.choice(
            np.arange(n), size=min(self.n_exemplars, n), replace=False
        ).tolist()

    def _precompute_closest_label_map(self):
        """
        Precompute exemplar indices for each target video based on label closeness.
        Returns a dict: {target_idx: [exemplar_indices]}
        """
        source_labels = np.array(
            [d[self.label_column] for d in self.source_db], dtype=float
        )
        target_labels = np.array(
            [d[self.label_column] for d in self.target_db], dtype=float
        )

        mapping = {}
        for i, t_label in enumerate(target_labels):
            diffs = np.abs(source_labels - float(t_label))
            sorted_indices = np.argsort(diffs)
            chosen = sorted_indices[: self.n_exemplars].tolist()
            mapping[i] = chosen

        return mapping
