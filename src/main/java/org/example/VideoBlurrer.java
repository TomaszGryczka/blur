package org.example;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.List;
import java.util.ArrayList;

public class VideoBlurrer {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        String inputVideoPath = "input.mp4";
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
        int chunkSize = 500; // Liczba klatek na porcję
        Mat frame = new Mat();

        while (true) {
            List<Mat> frames = new ArrayList<>();
            for (int i = 0; i < chunkSize; i++) {
                if (videoCapture.read(frame)) {
                    frames.add(frame.clone());
                } else {
                    break;
                }
            }

            if (frames.isEmpty()) {
                break;
            }

            BlurFramesTask task = new BlurFramesTask(frames, 0, frames.size());
            List<Mat> blurredFrames = pool.invoke(task);

            // Zapis przetworzonych klatek
            for (Mat blurredFrame : blurredFrames) {
                videoWriter.write(blurredFrame);
                blurredFrame.release();
            }

            // Zwolnij pamięć klatek
            for (Mat originalFrame : frames) {
                originalFrame.release();
            }
        }

        // Zwolnienie zasobów
        videoCapture.release();
        videoWriter.release();
        pool.shutdown();

        System.out.println("Video processing complete. Output saved to " + outputVideoPath);
    }

    static class BlurFramesTask extends RecursiveTask<List<Mat>> {
        private static final int THRESHOLD = 10; // Maksymalna liczba klatek do jednoczesnego przetwarzania
        private List<Mat> frames;
        private int start;
        private int end;

        public BlurFramesTask(List<Mat> frames, int start, int end) {
            this.frames = frames;
            this.start = start;
            this.end = end;
        }

        @Override
        protected List<Mat> compute() {
            if (end - start <= THRESHOLD) {
                // Przetwarzanie klatek w bieżącym zakresie
                List<Mat> blurredFrames = new ArrayList<>();
                for (int i = start; i < end; i++) {
                    Mat blurred = new Mat();
                    Imgproc.GaussianBlur(frames.get(i), blurred, new org.opencv.core.Size(15, 15), 0);
                    blurredFrames.add(blurred);
                }
                return blurredFrames;
            } else {
                // Podział na mniejsze zadania
                int mid = (start + end) / 2;
                BlurFramesTask task1 = new BlurFramesTask(frames, start, mid);
                BlurFramesTask task2 = new BlurFramesTask(frames, mid, end);

                invokeAll(task1, task2);

                List<Mat> result = new ArrayList<>();
                result.addAll(task1.join());
                result.addAll(task2.join());
                return result;
            }
        }
    }
}
