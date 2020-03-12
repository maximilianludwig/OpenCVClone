import glob, os, shutil, sys

if not os.path.exists("source_emotion"):
    print("source_emotion wurde nicht gefunden!")
    sys.exit(0)
if not os.path.exists("source_images"):
    print("source_images wurde nicht gefunden!")
    sys.exit(0)
if os.path.exists("sorted_set"):
    shutil.rmtree("sorted_set")

# setup destination directory
emotions = ["neutral","anger","contempt","disgust","fear","happy","sadness","surprise"]
os.makedirs("sorted_set")
for emotion in emotions: # create folder for each emotion
    os.makedirs("sorted_set/%s" % emotion)

participants = sorted(glob.glob("source_emotion/*"))  # Returns a list of all folders with participant numbers
for participant in participants:
    part = "%s" % participant[-4:]  # store current participant number
    sessions = sorted(glob.glob("%s/*" % participant))
    participant_neutral_handled = False
    for session in sessions:  # Store list of sessions for current participant
        files = sorted(glob.glob("%s/*" % session))
        for file in files:
            current_session = file[20:-30]
            file = open(file, 'r')
            # store emotion as int
            emotion = int(float(file.readline()))
            # get belonging emotion sequence
            images = sorted(glob.glob("source_images/%s/%s/*" % (part, current_session)))
            # copy last image from sequence to sorted set
            sourcefile_emotion = images[-1]  
            destination_emotional = "sorted_set/%s/%s" % (emotions[emotion], sourcefile_emotion[25:])
            shutil.copyfile(sourcefile_emotion,destination_emotional)
            # handle the neutral image only if not done for this participant before
            if not participant_neutral_handled:
                sourcefile_neutral = images[0]
                destination_neutral = "sorted_set/neutral/%s" % sourcefile_neutral[25:]
                shutil.copyfile(sourcefile_neutral,destination_neutral)
                participant_neutral_handled = True

print("sorted_set with emotion images created!")