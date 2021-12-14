package com.qualcomm.qti.snpe.imageclassifiers.thread;


import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.imageclassifiers.R;
import com.qualcomm.qti.snpe.imageclassifiers.detector.MobilenetDetector;
import com.qualcomm.qti.snpe.imageclassifiers.detector.TFMobilenetQuantizeDetector;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingDeque;

public class DetectorThread extends Thread{
    /** class's constance*/
    private static final String LOGTAG = DetectorThread.class.getSimpleName();
    private static final int MAX_QUEUE_SIZE = 20;
    private static final String AI_DETECTOR_THREAD = "AI detector thread";

    /** class main attribute */
    private Context mContext;
    private LinkedBlockingDeque<PreprocessResult> DetectorQueue;
    private PostProcessThread postProcessThread;
    private TFMobilenetQuantizeDetector mDetector;


    public DetectorThread (Application application,Context Context, PostProcessThread postProcessThread){
        this.DetectorQueue = new LinkedBlockingDeque<>();
        this.postProcessThread = postProcessThread;
        mContext = Context;
        this.mDetector = new TFMobilenetQuantizeDetector(mContext, application, R.raw.mobilenet_ssd_quantized);
        this.setName(AI_DETECTOR_THREAD);
    }
    @Override
    public void run() {
        while (true) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if (DetectorQueue.size() > 0) {
                try {
                    /**Execute model**/
                    PreprocessResult preprocessResult = DetectorQueue.takeFirst();
                    Bitmap original_dt = preprocessResult.getFrame();

                    long modelExecutionStart = System.currentTimeMillis();
                    final List<float[]> outputs = mDetector.detectFrame(preprocessResult);
                    long modelExecutionTime = System.currentTimeMillis() - modelExecutionStart;
                    postProcessThread.addItem(new DetectorResult(original_dt,outputs));
                    Log.d(LOGTAG, "model_Execute: " + modelExecutionTime);
                    /**Execute model**/
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }

        }
    }
    public boolean addItem(PreprocessResult preprocessResult) {
        if (DetectorQueue.size() > MAX_QUEUE_SIZE){
            try {
                DetectorQueue.takeFirst();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        return DetectorQueue.offerLast(preprocessResult);
    }
}
