import argparse
from pathlib import Path
from torcheval.metrics import PeakSignalNoiseRatio, StructuralSimilarity
import pytorch_ssim
import lpips
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torchvision.io import read_video
import warnings


# ===============================
# Metrics
# ===============================
class Metric(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def update(self, img1, img2):
        pass

    @abstractmethod
    def compute(self):
        pass


class PSNR(Metric):

    def __init__(self):
        super().__init__("PSNR")
        self.metric = PeakSignalNoiseRatio(device="cuda")

    def update(self, img1, img2):
        self.metric.update(img1, img2)

    def compute(self):
        return self.metric.compute().item()

class SSIM(Metric):
    def __init__(self):
        super().__init__("SSIM")
        self.values = []

    def update(self, img1, img2):
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        # normalize to -1 to 1
        img1 = img1 / 255.0 * 2 - 1
        img2 = img2 / 255.0 * 2 - 1

        score = pytorch_ssim.ssim(img1, img2)
        self.values.append(score)

    def compute(self):
        return torch.stack(self.values).mean().item()

class LPIPS(Metric):
    def __init__(self):
        super().__init__("LPIPS")
        self.metric = lpips.LPIPS(net='alex').cuda()
        self.values = []
    
    def update(self, img1, img2):
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        # normalize to -1 to 1
        img1 = img1 / 255.0 * 2 - 1
        img2 = img2 / 255.0 * 2 - 1

        score = self.metric(img1, img2)
        self.values.append(score)
    
    def compute(self):
        return torch.stack(self.values).mean().item()


# ===============================
# Main
# ===============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference-directory", type=str, required=True, help="The directory containing the reference videos")
    parser.add_argument("-g", "--generation-directory", type=str, required=True, help="The directory containing the generated videos")
    return parser.parse_args()
    

@torch.no_grad()
def main():
    args = parse_args()

    # make sure the reference and generation directories contain the same videos
    reference_dir = Path(args.reference_directory)
    generation_dir = Path(args.generation_directory)

    reference_videos = list(reference_dir.glob("*.mp4"))
    generation_videos = list(generation_dir.glob("*.mp4"))

    # sort the reference and generation videos
    reference_videos.sort()
    generation_videos.sort()

    # make sure the reference and generation directories contain the same videos
    if len(reference_videos) != len(generation_videos):
        warnings.warn(f"The reference and generation directories contain different number of videos: {len(reference_videos)} and {len(generation_videos)}")
    
    filtered_reference_videos = []
    filtered_generation_videos = []
    gen_video_stems = [video.stem for video in generation_videos]
    for reference_video in reference_videos:
        if reference_video.stem in gen_video_stems:
            filtered_reference_videos.append(reference_video)
            filtered_generation_videos.append(generation_videos[gen_video_stems.index(reference_video.stem)])

    # intialize the metrics
    metrics = [PSNR(), LPIPS(), SSIM()]

    # load the reference and generation videos
    for reference_video, generation_video in tqdm(zip(filtered_reference_videos, filtered_generation_videos), desc="Evaluating"):
        # read video
        assert reference_video.stem == generation_video.stem, f"The reference and generation videos have different names: {reference_video.stem} and {generation_video.stem}"
        ref_frames, _, _ = read_video(str(reference_video))
        gen_frames, _, _ = read_video(str(generation_video))

        # read video frame by frame
        for reference_frame, generation_frame in zip(ref_frames, gen_frames):
            # frame is in the shape of [H, W, C]
            reference_frame_tensor = reference_frame.cuda().permute(2, 0, 1)
            generation_frame_tensor = generation_frame.cuda().permute(2, 0, 1)

            # compute the metrics
            for metric in metrics:
                metric.update(reference_frame_tensor, generation_frame_tensor)
    
    # compute the metrics
    for metric in metrics:
        print(metric.name, metric.compute())    


if __name__ == "__main__":
    main()
    