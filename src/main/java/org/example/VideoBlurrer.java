package org.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;
import javax.imageio.ImageIO;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

public class VideoBlurrer {


    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String inputVideoPath = "input_cut.mp4";
        String outputVideoPath = "output_fork_join.avi";

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

        ForkJoinPool pool = new ForkJoinPool();
        Mat frame = new Mat();

        while (videoCapture.read(frame)) {
            Mat blurredFrame = pool.invoke(new BlurTask(frame));
            videoWriter.write(blurredFrame);
        }

        videoCapture.release();
        videoWriter.release();
        pool.shutdown();
        System.out.println("Video processing complete. Output saved to " + outputVideoPath);
    }

    static class BlurTask extends RecursiveTask<Mat> {
        private static final int THRESHOLD = 259200;
        private Mat frame;

        public BlurTask(Mat frame) {
            this.frame = frame;
        }

        @Override
        protected Mat compute() {
            if (frame.rows() * frame.cols() <= THRESHOLD) {
                Mat blurred = new Mat();
                Imgproc.GaussianBlur(frame, blurred, new org.opencv.core.Size(15, 15), 0);
                return blurred;
            } else {
                int midRow = frame.rows() / 2;
                Mat upperHalf = frame.rowRange(0, midRow);
                Mat lowerHalf = frame.rowRange(midRow, frame.rows());

                BlurTask upperTask = new BlurTask(upperHalf);
                BlurTask lowerTask = new BlurTask(lowerHalf);

                invokeAll(upperTask, lowerTask);

                Mat result = new Mat();
                Core.vconcat(List.of(upperTask.join(), lowerTask.join()), result);
                return result;
            }
        }
    }
}
