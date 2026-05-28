# =========================================================
# phone_detection.py
# =========================================================

import cv2
from ultralytics import YOLO

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# =========================================================
# SUPABASE
# =========================================================

supabase = create_client(

    SUPABASE_URL,

    SUPABASE_KEY

)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO("yolov8n.pt")

# =========================================================
# DETECT PHONE
# =========================================================

def detect_phone(

    frame,

    left,

    top,

    right,

    bottom,

    rollno,

    course_id,

    section

):

    try:

        results = model(frame)

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                label = model.names[cls]

                # =========================================
                # CELL PHONE FOUND
                # =========================================

                if label == "cell phone":

                    x1, y1, x2, y2 = map(

                        int,

                        box.xyxy[0]

                    )

                    # =====================================
                    # DRAW PHONE BOX
                    # =====================================

                    cv2.rectangle(

                        frame,

                        (x1, y1),

                        (x2, y2),

                        (0, 255, 255),

                        2

                    )

                    cv2.putText(

                        frame,

                        "PHONE",

                        (x1, y1 - 10),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 255, 255),

                        2

                    )

                    # =====================================
                    # CHECK OVERLAP WITH FACE
                    # =====================================

                    overlap = not (

                        right < x1 or
                        left > x2 or
                        bottom < y1 or
                        top > y2

                    )

                    if overlap:

                        # =================================
                        # UPDATE SESSION REPORT
                        # =================================

                        existing = supabase.table(

                            "session_report"

                        ).select("*").eq(

                            "s_rollno",

                            rollno

                        ).execute()

                        # =============================
                        # UPDATE EXISTING
                        # =============================

                        if existing.data:

                            already = existing.data[0]["s_phone"]

                            if already == 0:

                                supabase.table(

                                    "session_report"

                                ).update({

                                    "s_phone": 1

                                }).eq(

                                    "s_rollno",

                                    rollno

                                ).execute()

                        # =============================
                        # INSERT NEW
                        # =============================

                        else:

                            supabase.table(

                                "session_report"

                            ).insert({

                                "s_rollno": rollno,

                                "s_phone": 1,

                                "s_course_id": course_id,

                                "s_section": section

                            }).execute()

                        return True

        return False

    except Exception as e:

        print("Phone Detection Error:", e)

        return False