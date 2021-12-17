
#include "SortTracker.h"
//#include "FacePreprocess.h"
//#include "ImageEnhancement.h"
#include <string>
#include "jni.h"
#include "android/log.h"

using namespace SortTrackerNS;
//using namespace FacePreprocess;
//using namespace ImageEnhancement;
using namespace std;
using namespace cv;
using namespace std::chrono;
//using namespace bgslibrary::algorithms;

//const char* LOGTAG = "Native-lib";
extern const char* LOGTAG;

typedef struct _JNI_POSREC {
    jclass cls;
    jmethodID constructortorID;
    jfieldID frame;
    jfieldID id;
    jfieldID x1;
    jfieldID y1;
    jfieldID width;
    jfieldID height;
    jfieldID boxId;
    jfieldID update_hits;
    jfieldID predict_hits;
    jfieldID vx;
    jfieldID vy;
} JNI_POSREC;
//JNI_POSREC * jniPosRec = NULL;

JNI_POSREC * LoadJniPosRec(JNIEnv * env) {
    //if (jniPosRec != NULL) return;

    JNI_POSREC * jniPosRec = new JNI_POSREC;
    jniPosRec->cls = env->FindClass("com/qualcomm/qti/snpe/imageclassifiers/sortJni/TrackUtils$TrackResult");//"com/test/StudentRecord"

    if(jniPosRec->cls != NULL)
        printf("sucessfully created class");

    jniPosRec->constructortorID = env->GetMethodID(jniPosRec->cls, "<init>", "()V");
    if(jniPosRec->constructortorID != NULL){
        printf("sucessfully created ctorID");
    }

    jniPosRec->frame = env->GetFieldID(jniPosRec->cls, "frame", "I");//Ljava/lang/String;
    jniPosRec->id = env->GetFieldID(jniPosRec->cls, "id", "I");
    jniPosRec->x1 = env->GetFieldID(jniPosRec->cls, "x1", "F");
    jniPosRec->y1 = env->GetFieldID(jniPosRec->cls, "y1", "F");
    jniPosRec->width = env->GetFieldID(jniPosRec->cls, "width", "F");
    jniPosRec->height = env->GetFieldID(jniPosRec->cls, "height", "F");
    jniPosRec->boxId = env->GetFieldID(jniPosRec->cls, "boxId", "I");
    jniPosRec->update_hits = env->GetFieldID(jniPosRec->cls, "update_hits", "I");
    jniPosRec->predict_hits = env->GetFieldID(jniPosRec->cls, "predict_hits", "I");
    jniPosRec->vx = env->GetFieldID(jniPosRec->cls, "vx", "F");
    jniPosRec->vy = env->GetFieldID(jniPosRec->cls, "vy", "F");

    return jniPosRec;
}

extern "C" JNIEXPORT jlong JNICALL Java_com_qualcomm_qti_snpe_imageclassifiers_sortJni_nativeObjectTracker_initNative(JNIEnv *env, jclass clazzthis){
    return reinterpret_cast<jlong>(new SortTracker());
}

extern "C" JNIEXPORT jobjectArray JNICALL Java_com_qualcomm_qti_snpe_imageclassifiers_sortJni_nativeObjectTracker_nativeTrackSort(JNIEnv *env, jclass clazzthis,
                                                                                                                                            jobjectArray bboxesObjArr, jboolean isInited, jboolean isPredict, jlong cppSortTrackerPtr) {
    SortTracker *pSortTracker = reinterpret_cast<SortTracker *>(cppSortTrackerPtr);

    /*
    if (isInited && isPredict) {
        pSortTracker->doPredict();
        return NULL;
    }
     */

    JNI_POSREC *jniPosRec = LoadJniPosRec(env);
    //long start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();//SystemClock.uptimeMillis();

    //init bboxes data
    //vector<TrackingBox> detFrameData;
    vector<Rect_<float>> detRectData;

    if (!isInited || (!isPredict)) {
        //Mat& frameCv = *(Mat*) frameCvPtr;
        int len1 = env->GetArrayLength(bboxesObjArr);
        jfloatArray dim = (jfloatArray) env->GetObjectArrayElement(bboxesObjArr, 0);
        int len2 = env->GetArrayLength(dim);

        for (int i = 0; i < len1; ++i) {//number of bboxes
            jfloatArray oneDim = (jfloatArray) env->GetObjectArrayElement(bboxesObjArr, i);
            jfloat *element = env->GetFloatArrayElements(oneDim, 0);
            //allocate localArray[i] using len2 //x1, y1, x2, y2 (len2 = 4)
            float x1 = element[0], y1 = element[1], x2 = element[2], y2 = element[3];
            Rect_<float> rect0(x1, y1, x2 - x1, y2 - y1);
            detRectData.push_back(rect0);
        }
    }

    //run doSortTrack
    vector<TrackingBox> trackResults =  pSortTracker->doSortTrack(detRectData);//detFrameData);

    //process result to return list
    jobjectArray jPosRecArray = env->NewObjectArray(trackResults.size(), jniPosRec->cls, NULL);
    for (size_t i = 0; i < trackResults.size(); i++) {
        jobject jPosRec = env->NewObject(jniPosRec->cls, jniPosRec->constructortorID);
        //fill data
        env->SetIntField(jPosRec, jniPosRec->frame, trackResults[i].frame);//frame
        env->SetIntField(jPosRec, jniPosRec->id, trackResults[i].id);
        env->SetFloatField( jPosRec, jniPosRec->x1, trackResults[i].box.x);
        env->SetFloatField( jPosRec, jniPosRec->y1, trackResults[i].box.y);
        env->SetFloatField( jPosRec, jniPosRec->width, trackResults[i].box.width);
        env->SetFloatField( jPosRec, jniPosRec->height, trackResults[i].box.height);
        env->SetIntField( jPosRec, jniPosRec->boxId, trackResults[i].boxId);
        env->SetIntField( jPosRec, jniPosRec->update_hits, trackResults[i].update_hits);
        env->SetIntField( jPosRec, jniPosRec->predict_hits, trackResults[i].predict_hits);
        env->SetFloatField( jPosRec, jniPosRec->vx, trackResults[i].vx);
        env->SetFloatField( jPosRec, jniPosRec->vy, trackResults[i].vy);
        //
        env->SetObjectArrayElement(jPosRecArray, i, jPosRec);
    }

    return jPosRecArray;
}

extern "C" JNIEXPORT void JNICALL Java_com_qualcomm_qti_snpe_imageclassifiers_sortJni_nativeObjectTracker_releaseNative(JNIEnv *env, jclass clazzthis,
                                                                                                                                  jlong cppSortTrackerPtr)
{
    SortTracker* pSortTracker = reinterpret_cast<SortTracker*>(cppSortTrackerPtr);
    delete pSortTracker;
}
