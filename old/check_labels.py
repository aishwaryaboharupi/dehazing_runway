import os
import cv2
import numpy as np

clean_dir = "clean_images"
label_dir = "label"

if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
    print(f"\n[!] The '{label_dir}' folder is empty. Please ensure your 400 shape masks are inside it.")
else:
    
    clean_dict = {os.path.splitext(f)[0]: f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    label_dict = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
    
    
    matched_pairs = []
    for lbl_base, lbl_file in label_dict.items():
        if lbl_base in clean_dict:
            matched_pairs.append((clean_dict[lbl_base], lbl_file))
        else:
           
            for cln_base, cln_file in clean_dict.items():
                if cln_base in lbl_base or lbl_base in cln_base:
                    matched_pairs.append((cln_file, lbl_file))
                    break

    print("\n=========================================")
    print("        DATASET PAIRING STATUS           ")
    print("=========================================")
    print(f" Total Runway Images Found : {len(clean_dict)}")
    print(f" Total Hand-Picked Shapes  : {len(label_dict)}")
    print(f" Successfully Paired Units : {len(matched_pairs)} pairs")
    print("=========================================")

    if matched_pairs:
       
        cln_name, lbl_name = matched_pairs[0]
        img = cv2.imread(os.path.join(clean_dir, cln_name))
        mask = cv2.imread(os.path.join(label_dir, lbl_name))
        
        img_resized = cv2.resize(img, (640, 480))
        mask_resized = cv2.resize(mask, (640, 480))
        
        
        overlay = img_resized.copy()
        overlay[np.where((mask_resized > 200).all(axis=2))] = [0, 0, 255] 
        
        preview = np.hstack((img_resized, mask_resized, overlay))
        cv2.imwrite("runway_mask_verification.png", preview)
        print(f"\n[SUCCESS] Linked pair verified: Image ({cln_name}) <--> Shape ({lbl_name})")
        print(" Open 'runway_mask_verification.png' to see the red target overlay layout!")
    else:
        print("\n[INFO] No overlap found using cross-string matching.")
        print(" Let's print a sample to see how the numbers/letters differ:")
        print(f" Sample Clean Image Name : {list(clean_dict.values())[0] if clean_dict else 'None'}")
        print(f" Sample Shape Mask Name  : {list(label_dict.values())[0] if label_dict else 'None'}")