import cv2
import matplotlib.pyplot as plt
import torch
import os
import h5py
from matplotlib import gridspec
from Utils.Utils import denormalize

def save_figure(fig, filename="comparison_plot.png"):
    """
    Save the figure to a specified file.
    """
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved figure as {filename}")


class TestImageLoader:
    def __init__(self, base_dir):
        """
        This is add-on class for MPI-INF-3DHP Dataset if you want to visualize both joints and images for comparison on their test set.
        Please make sure your folder that contains test set have a similar structure as original structure of the test set:
        TS1, TS2, ..., TS6.
        Parameters:
        - base_dir (str): Path to the main directory containing subdirectories TS1, TS2, ..., TS6.
        """
        self.base_dir = base_dir
        self.sub_dirs = [f"TS{i}" for i in range(1, 7)]  # Folders TS1 to TS6
        self.image_paths = []
        self.sequence_ids = []
        self.indexes = []
        self.joints_2D = []  # List to store joint2D coordinates

        # Load all image paths and 2D joints in the correct order
        self._load_image_paths()

    def _load_mat_file(self, file_path):
        """
        Load data from a .mat file, supporting both v7.3 and older formats.

        Parameters:
        - file_path (str): Path to the .mat file.

        Returns:
        - valid_frames (numpy array): Array of valid frame flags.
        - joint2D (numpy array): Array of joint2D coordinates for each valid frame.
        """
        if file_path.endswith('.mat'):
            try:
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                valid_frames = mat_data.get('valid_frame', []).flatten().astype(bool)
                joint2D = mat_data.get('annot2', [])  # Assuming joint2D is stored in this field
                return valid_frames, joint2D
            except NotImplementedError:
                with h5py.File(file_path, 'r') as f:
                    valid_frames = f['valid_frame'][()].astype(bool)
                    joint2D = f['annot2'][()]  # Assuming joint2D is stored in this field
                    return valid_frames, joint2D
        else:
            raise ValueError("Unsupported file format. Expected a .mat file.")

    def _load_image_paths(self):
        """
        Load image paths and 2D joint data from subdirectories in order (TS1, TS2, ..., TS6).
        """
        sequence_id = 0

        for idx, sub_dir in enumerate(self.sub_dirs):
            annot_path = os.path.join(self.base_dir, sub_dir, "annot_data.mat")
            if not os.path.exists(annot_path):
                print(f"Annotation file not found in {sub_dir}, skipping...")
                continue

            valid_frames, joint2D = self._load_mat_file(annot_path)

            img_dir = os.path.join(self.base_dir, sub_dir, "imageSequence")
            if not os.path.exists(img_dir):
                print(f"Image sequence directory not found in {sub_dir}, skipping...")
                continue

            images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

            # Adjust valid_frames and joint2D length to match images if needed
            min_length = min(len(valid_frames), len(images))
            valid_frames = valid_frames[:min_length]
            images = images[:min_length]
            joint2D = joint2D[:min_length]

            # Load only the valid frames
            for i, img_name in enumerate(images):
                if valid_frames[i]:
                    img_path = os.path.join(img_dir, img_name)
                    self.image_paths.append(img_path)
                    self.sequence_ids.append(sequence_id)
                    self.indexes.append(idx + 1)
                    self.joints_2D.append(joint2D[i])  # Store 2D joint data for each valid frame
                    sequence_id += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get image, joint2D, and metadata by index.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - image (numpy array): Loaded image.
        - joint2D (numpy array): 2D joint coordinates.
        - sequence_id (int): Unique sequence ID for the image.
        - ts_idx (int): Index representing which TS folder the image belongs to (1 for TS1, 2 for TS2, etc.).
        """
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        joint2D = self.joints_2D[idx]
        sequence_id = self.sequence_ids[idx]
        ts_idx = self.indexes[idx]

        return image, joint2D, sequence_id, ts_idx


def visualize_comparison_3_model(model, model_1, model_2, test_loader, image_loader, sequence_idx, joints_left,
                                 joints_right, skeleton_pairs, elev=20, azim=30):
    """
    This code used for models visualization and comparison on MPI-INF-3DHP Dataset's test set.
    Default settings are set for Our model, MotionAGFormer-L, and MotionAGFormer-B with
    2D joint input (T, 17, 3) with 3 Dimensions (x,y,confident score). If you want to use for
    comparing other models, please modify the input.
    MotionAGFormer weights for MPI-INF-3DHP: https://github.com/TaatiTeam/MotionAGFormer.git

    Default parameters in tested process:
    - Elev: 280
    - Azim: 270
    - Joints left: [5, 6, 7, 11, 12, 13]
    - Joints right: [2, 3, 4, 8, 9, 10]
    - Sequence indexes: There are 2875 valid frames from TS1 to TS6 in the test set,
    please enter a number between 0 and 2874 to select a frame to compare
    TS1: 0 to 602
    TS2: 603 to 1142
    TS3: 1143 to 1648
    TS4: 1649 to 2206
    TS5: 2207 to 2482
    TS6: 2483 to 2874
    """

    model.eval()
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        data = list(test_loader)[sequence_idx]
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        input_2D = input_2D.float().to(next(model.parameters()).device)

        image, joint2D, sequence_id, ts_idx = image_loader[sequence_idx]

        if joint2D.shape[0] == 1 and joint2D.shape[1:] == (17, 2):
            joint2D = joint2D.reshape(17, 2)

        output_3D, _ = model(input_2D)
        output_3D = output_3D * scale.to(output_3D.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pad = (output_3D.size(1) - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)

        output_3D_model_1 = model_1(input_2D)
        output_3D_model_1 = output_3D_model_1 * scale.to(output_3D_model_1.device).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)
        pred_out_model_1 = output_3D_model_1[:, pad].unsqueeze(1)

        output_3D_model_2 = model_2(input_2D)
        output_3D_model_2 = output_3D_model_2 * scale.to(output_3D_model_2.device).unsqueeze(-1).unsqueeze(
            -1).unsqueeze(-1)
        pred_out_model_2 = output_3D_model_2[:, pad].unsqueeze(1)

        pred_out[..., 14, :] = 0
        pred_out_model_1[..., 14, :] = 0
        pred_out_model_2[..., 14, :] = 0
        gt_3D[..., 14, :] = 0

        pred_out = denormalize(pred_out, seq)
        pred_out_model_1 = denormalize(pred_out_model_1, seq)
        pred_out_model_2 = denormalize(pred_out_model_2, seq)

        pred_out = pred_out - pred_out[..., 14:15, :]
        pred_out_model_1 = pred_out_model_1 - pred_out_model_1[..., 14:15, :]
        pred_out_model_2 = pred_out_model_2 - pred_out_model_2[..., 14:15, :]
        out_target = gt_3D - gt_3D[..., 14:15, :]

        fig = plt.figure(figsize=(40, 20))

        # Create a grid with gridspec
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 2, 2, 2], wspace=0.3)

        # Small image plot on the left
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_img.axis("off")
        ax_img.set_title(f"TS{ts_idx}, Sequence ID: {sequence_id}")

        for (start, end) in skeleton_pairs:
            ax_img.plot(
                [joint2D[start, 0], joint2D[end, 0]],
                [joint2D[start, 1], joint2D[end, 1]],
                color='blue' if start in joints_left or end in joints_left else 'red', linewidth=2
            )

        for j in range(joint2D.shape[0]):
            ax_img.scatter(joint2D[j, 0], joint2D[j, 1], color='green', s=10)

        # AS_AGFormer Model 3D Joint Visualization (middle)
        ax1 = fig.add_subplot(gs[1], projection='3d')
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_title("AS-AGFormer")
        for (start, end) in skeleton_pairs:
            ax1.plot(
                [out_target[0, 0, start, 0], out_target[0, 0, end, 0]],
                [out_target[0, 0, start, 1], out_target[0, 0, end, 1]],
                [out_target[0, 0, start, 2], out_target[0, 0, end, 2]],
                color='black', alpha=0.5, linewidth=1.5
            )

        for (start, end) in skeleton_pairs:
            color = 'red' if start in joints_right or end in joints_right else 'blue'
            ax1.plot(
                [pred_out[0, 0, start, 0], pred_out[0, 0, end, 0]],
                [pred_out[0, 0, start, 1], pred_out[0, 0, end, 1]],
                [pred_out[0, 0, start, 2], pred_out[0, 0, end, 2]],
                color=color, linewidth=2
            )

        for j in range(pred_out.shape[2]):
            color = 'red' if j in joints_right else 'blue'
            ax1.scatter(pred_out[0, 0, j, 0], pred_out[0, 0, j, 1], pred_out[0, 0, j, 2], color=color, s=30)

        # MotionAGFormer-L Model 3D Joint Visualization (right)
        ax2 = fig.add_subplot(gs[2], projection='3d')
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_title("MotionAGFormer-L (26 layers)")
        for (start, end) in skeleton_pairs:
            ax2.plot(
                [out_target[0, 0, start, 0], out_target[0, 0, end, 0]],
                [out_target[0, 0, start, 1], out_target[0, 0, end, 1]],
                [out_target[0, 0, start, 2], out_target[0, 0, end, 2]],
                color='black', alpha=0.5, linewidth=1.5
            )

        for (start, end) in skeleton_pairs:
            color = 'red' if start in joints_right or end in joints_right else 'blue'
            ax2.plot(
                [pred_out_model_1[0, 0, start, 0], pred_out_model_1[0, 0, end, 0]],
                [pred_out_model_1[0, 0, start, 1], pred_out_model_1[0, 0, end, 1]],
                [pred_out_model_1[0, 0, start, 2], pred_out_model_1[0, 0, end, 2]],
                color=color, linewidth=2
            )

        for j in range(pred_out_model_1.shape[2]):
            color = 'red' if j in joints_right else 'blue'
            ax2.scatter(pred_out_model_1[0, 0, j, 0], pred_out_model_1[0, 0, j, 1], pred_out_model_1[0, 0, j, 2],
                        color=color, s=30)

        # MotionAGFormer-L Model 3D Joint Visualization (right)
        ax3 = fig.add_subplot(gs[3], projection='3d')
        ax3.view_init(elev=elev, azim=azim)
        ax3.set_title("MotionAGFormer-B (16 layers)")
        for (start, end) in skeleton_pairs:
            ax3.plot(
                [out_target[0, 0, start, 0], out_target[0, 0, end, 0]],
                [out_target[0, 0, start, 1], out_target[0, 0, end, 1]],
                [out_target[0, 0, start, 2], out_target[0, 0, end, 2]],
                color='black', alpha=0.5, linewidth=1.5
            )

        for (start, end) in skeleton_pairs:
            color = 'red' if start in joints_right or end in joints_right else 'blue'
            ax3.plot(
                [pred_out_model_2[0, 0, start, 0], pred_out_model_2[0, 0, end, 0]],
                [pred_out_model_2[0, 0, start, 1], pred_out_model_2[0, 0, end, 1]],
                [pred_out_model_2[0, 0, start, 2], pred_out_model_2[0, 0, end, 2]],
                color=color, linewidth=2
            )

        for j in range(pred_out_model_1.shape[2]):
            color = 'red' if j in joints_right else 'blue'
            ax3.scatter(pred_out_model_2[0, 0, j, 0], pred_out_model_2[0, 0, j, 1], pred_out_model_2[0, 0, j, 2],
                        color=color, s=30)

        black_line = ax1.plot([], [], [], color="black", alpha=0.5, linewidth=1.5, label="Ground Truth 3D")[0]
        red_line = ax1.plot([], [], [], color="red", linewidth=2, label="Predicted Right Joints")[0]
        blue_line = ax1.plot([], [], [], color="blue", linewidth=2, label="Predicted Left Joints")[0]
        ax1.legend(handles=[black_line, red_line, blue_line])
        ax2.legend(handles=[black_line, red_line, blue_line])
        ax3.legend(handles=[black_line, red_line, blue_line])

        plt.show()

        save_figure(fig, filename="image.png")