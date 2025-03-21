import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import random


if __name__ == "__main__":
    
    db = "unovis"

    if db == "mit": 

        with open('dictionaries/mit_noisy_ecg_by_patients.pkl', 'rb') as f:
            mit_noisy_ecg = pickle.load(f)

        with open('dictionaries/mit_noisy_ecg_labels_by_patients.pkl', 'rb') as f:
            mit_noisy_ecg_labels = pickle.load(f)
        
        with open('dictionaries/mit_reference_ecg_by_patients.pkl', 'rb') as f:
            mit_reference_ecg = pickle.load(f)
        
        plotted_usable = False
        plotted_unusable = False
        while (plotted_unusable and plotted_usable) == False:
                
            rand_patient_key = random.randrange(100000, 234000, 1000)
            rand_segment_idx = random.randrange(0,1500)

                
            try:
            # print(mit_noisy_ecg_labels[rand_patient_key][rand_segment_idx])

                if mit_noisy_ecg_labels[rand_patient_key][rand_segment_idx] == 0 and plotted_unusable == False:
                    
                    fig = plt.figure(facecolor="y")
                    plt.plot(mit_noisy_ecg[rand_patient_key][rand_segment_idx, :], label="unusable segment")
                    plt.plot(mit_reference_ecg[rand_patient_key][rand_segment_idx, :] , label="reference segment" )
                    plt.legend(loc="best")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.title("Example Unusable Segment from MIT Database")
                    plt.savefig ("sample_segments/mit_unusable_segment.png")
                    plt.clf()

                    plotted_unusable = True
                    print("plotted_unusable")

                elif mit_noisy_ecg_labels[rand_patient_key][rand_segment_idx] == 1 and plotted_usable == False:
                    
                    fig = plt.figure(facecolor="g")
                    plt.plot(mit_noisy_ecg[rand_patient_key][rand_segment_idx, :], label="usable segment")
                    plt.plot(mit_reference_ecg[rand_patient_key][rand_segment_idx, :] , label="reference segment" )
                    plt.legend(loc="best")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.title("Example Usable Segment from MIT Database")
                    plt.savefig ("sample_segments/mit_usable_segment.png")
                    plt.clf()

                    plotted_usable = True
                    print("plotted_usable")
            except:
                pass

    elif db == "unovis":

        with open('dictionaries/final_dicts_1703/unovis_cecg_by_patients.pkl', 'rb') as f:
            unovis_noisy_ecg = pickle.load(f)

        with open('dictionaries/final_dicts_1703/unovis_cecg_labels_by_patients.pkl', 'rb') as f:
            unovis_noisy_ecg_labels = pickle.load(f)

        with open('dictionaries/final_dicts_1703/unovis_reference_ecg_by_patients.pkl', 'rb') as f:
            unovis_reference_ecg = pickle.load(f)
        
        plotted_usable = False
        plotted_unusable = False
        while (plotted_unusable and plotted_usable) == False:
                
            rand_patient_key = random.randrange(50, 200)
            rand_segment_idx = random.randrange(0,1500)

                
            try:
            # print(unovis_noisy_ecg_labels[rand_patient_key][rand_segment_idx])

                if unovis_noisy_ecg_labels[rand_patient_key][rand_segment_idx] == 0 and plotted_unusable == False:
                    
                    fig = plt.figure(facecolor="y")
                    plt.plot(unovis_noisy_ecg[rand_patient_key][rand_segment_idx, :], label="unusable segment")
                    plt.plot(unovis_reference_ecg[rand_patient_key][rand_segment_idx, :] , label="reference segment" )
                    plt.legend(loc="best")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.title("Example Unusable Segment from UnoViS Database")
                    plt.savefig ("sample_segments/unovis_unusable_segment.png")
                    plt.clf()

                    plotted_unusable = True
                    print("plotted_unusable")

                elif unovis_noisy_ecg_labels[rand_patient_key][rand_segment_idx] == 1 and plotted_usable == False:
                    
                    fig = plt.figure(facecolor="g")
                    plt.plot(unovis_noisy_ecg[rand_patient_key][rand_segment_idx, :], label="usable segment")
                    plt.plot(unovis_reference_ecg[rand_patient_key][rand_segment_idx, :] , label="reference segment" )
                    plt.legend(loc="best")
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.title("Example Usable Segment from UnoViS Database")
                    plt.savefig ("sample_segments/unovis_usable_segment.png")
                    plt.clf()

                    plotted_usable = True
                    print("plotted_usable")
            except:
                pass