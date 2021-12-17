package com.qualcomm.qti.snpe.imageclassifiers.sortJni;

public class TrackUtils {
    public class TrackResult {
        public int    frame;
        public int    id;
        public float  x1;
        public float  y1;
        public float  width;
        public float  height;
        public int    boxId;
        //other
        public int    update_hits;
        public int    predict_hits;

        public float vx;
        public float vy;
    }

    static final String LOGTAG = TrackUtils.class.getSimpleName();

    //public static native TrackResult[] nativeTrackSort(float[][] bboxesObjArr);
    /*
    public static TrackResult[] prepareTrackSort(List<Bbox> bboxes){
        float[][] bboxesObjArr = new float[bboxes.size()][4];

        for (int i = 0; i < bboxes.size(); i++) {
            Bbox box = bboxes.get(i);
            bboxesObjArr[i] = new float[]{box.x1, box.y1, box.x2, box.y2};
        }

        TrackResult[] trResults = nativeTrackSort(bboxesObjArr);

        Log.d(LOGTAG, "prepareTrackSort " + trResults.length);

        return trResults;
    }
     */

    /*
        new RectF(x1, y1, x2, y2);//left, top, right, bottom
    */
    /*
    public static TrackResult[] prepareTrackSortHuman(List<Recognition> bboxes){
        float[][] bboxesObjArr = new float[bboxes.size()][4];

        for (int i = 0; i < bboxes.size(); i++) {
            //Recognition box = bboxes.get(i);
            RectF loc = bboxes.get(i).getLocation();
            bboxesObjArr[i] = new float[]{loc.left, loc.top, loc.right, loc.bottom};
        }

        TrackResult[] trResults = nativeTrackSort(bboxesObjArr);

        //Log.d(LOGTAG, "prepareTrackSort " + trResults.length);

        return trResults;
    }
     */
}
