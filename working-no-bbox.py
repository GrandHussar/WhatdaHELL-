from roboflow import Roboflow
import cv2
import numpy as np
import json
import supervision as sv

api_key = "lhqfUB5mFhHzkombn0qo"
rf = Roboflow(api_key=api_key)

workspace_name = "zzz-x9kiy"
project_name = "ambulance-detection-m9kn3"
workspace = rf.workspace(workspace_name)
project = workspace.project(project_name)
latest_version = max(project.versions(), key=lambda v: v.id)

if latest_version is None:
    print("Error: No versions found for the project.")
else:
    model = latest_version.model
    if model is None:
        print("Error: No model found for the latest version.")
    else:
        video_path = "C://Users//Luigi T. Francisco//Desktop//Productivity//Nigger//emtech-fnls//ambulance2.webm"  # replace with proper dir pls
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            response = model.predict(frame, confidence=80, overlap=80).json()
            

            if response is not None:
                alpha=response
                
                
 
                labels = [item["class"] for item in alpha["predictions"]]
                
                
                
                
                detections = sv.Detections.from_roboflow(alpha)
                label_annotator = sv.LabelAnnotator()
                bounding_box_annotator = sv.BoxAnnotator()
                a=detections.xyxy
                print(a)
                x1 = int(a[0][0])
                x2 = int(a[0][2])
                y1 = int(a[0][1])
                y2 = int(a[0][3])
                print(x1,x2,y1,y2)
                
            
                

                image = frame
                
                
       
                annotated_image = bounding_box_annotator.annotate(
                    scene=image, detections=detections,skip_label=True)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame, "Ambulance", (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), )

                
                
                

                
                img = image

   
                cv2.imshow("Frame", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("Error: Failed to get a response from the model.")
                
        cap.release()
        cv2.destroyAllWindows()
