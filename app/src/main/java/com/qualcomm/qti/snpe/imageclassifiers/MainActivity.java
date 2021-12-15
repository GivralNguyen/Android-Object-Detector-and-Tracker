/*
 * Copyright (c) 2016 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers;
import static org.bytedeco.javacpp.Loader.getCacheDir;


import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Bbox;
import com.qualcomm.qti.snpe.imageclassifiers.detector.TFMobilenetQuantizeDetector;
import com.qualcomm.qti.snpe.imageclassifiers.detector.Recognition;
import com.qualcomm.qti.snpe.imageclassifiers.thread.DetectorThread;
import com.qualcomm.qti.snpe.imageclassifiers.thread.FrameLoaderResult;
import com.qualcomm.qti.snpe.imageclassifiers.thread.PostProcessThread;
import com.qualcomm.qti.snpe.imageclassifiers.thread.PreprocessThread;

import com.qualcomm.qti.snpe.imageclassifiers.view.VideoSurfaceView;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_java;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MainActivity extends Activity {
    /** Declare variables **/
    private static final String LOGTAG = MainActivity.class.getSimpleName();
    private static final int SELECT_VIDEO = 1;
    private Bitmap testImageBitmap;
    private MediaMetadataRetriever mRetriever;

    /** Declare variables **/

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Loader.load(opencv_java.class);
        setContentView(R.layout.activity_main);
        long loadBmpStart = System.currentTimeMillis();
        testImageBitmap = loadBmpImage(R.raw.image2);/**Load bitmap image**/
        long loadBmpTime = System.currentTimeMillis()- loadBmpStart;
        Log.d(LOGTAG,"loadBmpTime_time: "+ loadBmpTime);
        final PostProcessThread postProcessThread = new PostProcessThread(this);
        postProcessThread.setImageView(this.<VideoSurfaceView>findViewById(R.id.surfaceView));
        final DetectorThread detectorThread = new DetectorThread(this.getApplication(),this,postProcessThread);
        final PreprocessThread preprocessThread = new PreprocessThread(detectorThread);

        preprocessThread.start();
        detectorThread.start();
        postProcessThread.start();
        Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                pickVideo();
            }
        });
        final Button button2 = (Button) findViewById(R.id.button2);
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                preprocessThread.addVideo(mRetriever);
            }
        });
    }

    private void pickVideo(){
        Intent i = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, SELECT_VIDEO);
    }

    private void onVideoPicked(Intent data){
        Uri uri = data.getData();
        Log.e(LOGTAG, "uri = " + uri);
        mRetriever = new MediaMetadataRetriever();
        mRetriever.setDataSource(this, uri);
        Bitmap bitmap = mRetriever.getFrameAtTime(0);
        ImageView imageView = (ImageView) findViewById(R.id.imageView);
        imageView.setImageBitmap(bitmap);
    }

    @ Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_VIDEO) {
                onVideoPicked(data);
            }
        }
    }

    private Bitmap loadBmpImage(int Input){
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inScaled = false;
        Bitmap testBmp = BitmapFactory.decodeResource(getResources(),Input,o);

        return testBmp;
    }



}
