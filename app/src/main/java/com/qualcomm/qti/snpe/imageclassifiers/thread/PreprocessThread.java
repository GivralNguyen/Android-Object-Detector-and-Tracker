package com.qualcomm.qti.snpe.imageclassifiers.thread;

import static com.qualcomm.qti.snpe.imageclassifiers.detector.MobilenetDetector.MODEL_HEIGHT;
import static com.qualcomm.qti.snpe.imageclassifiers.detector.MobilenetDetector.MODEL_WIDTH;

import android.annotation.TargetApi;
import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.MediaMetadataRetriever;
import android.os.Build;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.imageclassifiers.R;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.MobilenetDetector;
import com.qualcomm.qti.snpe.imageclassifiers.detector.TFMobilenetQuantizeDetector;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingDeque;

public class PreprocessThread extends Thread{
    /** class's constance*/
    private static final String LOGTAG = PreprocessThread.class.getSimpleName();
    private static final int MAX_QUEUE_SIZE = 20;
    private static final String PREPROCESSING_THREAD = "AI preprocessing thread";
    private DetectorThread DetectorThread;

    /** class main attribute */
    private LinkedBlockingDeque<FrameLoaderResult> preprocessQueue;
    private boolean isProcess = true;

    private float[] inputValues = new float[MODEL_WIDTH * MODEL_HEIGHT * 3];
    private FloatTensor inputTensor = null;
    private Map<String, FloatTensor> inputs = new HashMap<>();
    String mInputLayer = "";
    private MediaMetadataRetriever mRetriever;


    /** Constructor 2: using default detector */
    public PreprocessThread( DetectorThread mDetectorThread) {
        this.DetectorThread = mDetectorThread;
        this.preprocessQueue = new LinkedBlockingDeque<>();
        this.setName(PREPROCESSING_THREAD);
    }
    /** do preprocessing */
    @TargetApi(Build.VERSION_CODES.P)
    @Override
    public void run() {
        while (isProcess) {
            /** need sleep to cpu scheduling*/
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if(mRetriever != null) {
                int i = 0;
                while(true) {
                    /** Loop running detection **/
                    try {
                        Bitmap bitmap = mRetriever.getFrameAtIndex(i++);
                        preprocessFrame(new FrameLoaderResult(bitmap, i));
                    } catch (Exception e){
                        Log.e(LOGTAG, "Running out frame, total = " + i);
                        mRetriever = null;
                        break;
                    }
                }
            }

            if (preprocessQueue.size() > 0){
                try {
                    FrameLoaderResult frameLoader = preprocessQueue.takeFirst();
                    preprocessFrame(frameLoader);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

        }
    }

    private void preprocessFrame(FrameLoaderResult frameLoader){
        /**Preprocessing**/
        long preProcessStart = System.currentTimeMillis();
        /**Resizing Bitmap**/
        Bitmap original = frameLoader.getFrame();
        int frame_id_preprocess = frameLoader.getFrame_id_loader();
        final float scaleX = MODEL_WIDTH / (float) (original.getWidth());
        final float scaleY = MODEL_HEIGHT / (float) (original.getHeight());
        final Matrix scalingMatrix = new Matrix();
        scalingMatrix.postScale(scaleX, scaleY);
        Bitmap resizedBitmap = Bitmap.createBitmap(original,
                0, 0,
                original.getWidth(), original.getHeight(),scalingMatrix,false);
        Mat frameCv = new Mat();//frame.getWidth(), frame.getHeight(), CvType.CV_8UC3);
        Bitmap frame32 = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true);//ismutable
        Utils.bitmapToMat(frame32, frameCv);//frame32, frameCv);
        Imgproc.cvtColor(frameCv , frameCv , Imgproc.COLOR_RGBA2BGR);//COLOR_RGBA2RGB
        frameCv.convertTo(frameCv, CvType.CV_32F);//, 1.0, 0); //convert to 32F
        Core.subtract(frameCv, new Scalar(128.0f, 128.0f, 128.0f), frameCv);
        Core.divide(frameCv, new Scalar(128.0f, 128.0f, 128.0f), frameCv);
        frameCv.get(0, 0, inputValues); //image.astype(np.float32)
        /**Resizing Bitmap**/

        DetectorThread.addItem(new PreprocessResult(original,inputValues,frame_id_preprocess));
        long preProcessTime = System.currentTimeMillis()- preProcessStart;
        Log.d(LOGTAG,"Preprocess_time: "+ preProcessTime);
        /**Preprocessing**/
    }

    public boolean addItem(FrameLoaderResult frame) {
        if (preprocessQueue.size() > MAX_QUEUE_SIZE){
            try {
                preprocessQueue.takeFirst();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return preprocessQueue.offerLast(frame);
    }


    public void addVideo(MediaMetadataRetriever retriever) {
        mRetriever = retriever;
    }
}
