import os
import cv2

frames_dir = r"C:\College\06_BENGAWAN\blender\satu\rendered_frames"
output_video_path = r"C:\College\06_BENGAWAN\blender\satu\code\input_rendered.mp4"

fps = 24  # Updated to 24 FPS
width, height = 1920, 1080

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
for i in range(1, 61):
    img_path = os.path.join(frames_dir, f"{i:04d}.png")
    if os.path.exists(img_path):
        frame = cv2.imread(img_path)
        out.write(frame)
        frame_count += 1

out.release()
print(f"Successfully created video: {output_video_path} at {fps} FPS with {frame_count} frames.")
