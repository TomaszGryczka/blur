package org.example;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

public class SingleThreadBlurrer {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String inputVideoPath = "input.mp4";
        String outputVideoPath = "output_single_thread.avi";

        VideoCapture videoCapture = new VideoCapture(inputVideoPath);
        if (!videoCapture.isOpened()) {
            System.err.println("Failed to open video file: " + inputVideoPath);
            return;
        }

        int frameWidth = (int) videoCapture.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int frameHeight = (int) videoCapture.get(Videoio.CAP_PROP_FRAME_HEIGHT);
        int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');
        int fps = (int) videoCapture.get(Videoio.CAP_PROP_FPS);
        VideoWriter videoWriter = new VideoWriter(outputVideoPath, fourcc, fps, new org.opencv.core.Size(frameWidth, frameHeight));

        if (!videoWriter.isOpened()) {
            System.err.println("Failed to open video writer: " + outputVideoPath);
            return;
        }

        Mat frame = new Mat();

        while (videoCapture.read(frame)) {
            Mat blurredFrame = new Mat();
            Imgproc.GaussianBlur(frame, blurredFrame, new org.opencv.core.Size(15, 15), 0);
            videoWriter.write(blurredFrame);
        }

        videoCapture.release();
        videoWriter.release();

        System.out.println("Video processing complete. Output saved to " + outputVideoPath);
    }
}
