import pyaudio
import wave
import numpy as np
from pynput import keyboard
import time
def EnergyPerSampleInDecibel(audioframe):
    audio_data = np.frombuffer(audioframe, dtype=np.short)
    decibels = np.max(audio_data)
    return decibels

def classifyFrame(audioframe, background, forgetfactor, adjustment, threshold, level):
    current = EnergyPerSampleInDecibel(audioframe)
    isSpeech = False
    level = ((level * forgetfactor) + current) / (forgetfactor + 1)
    if level < background:
        level = background
    if (level - background > threshold):
        isSpeech = True
    if not isSpeech:
        if current < background:
            background = current
        else:
            background += (current - background) * adjustment
    return isSpeech, background, level

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    WAVE_OUTPUT_FILENAME = "#Your Path"
    # SILENCE_THRESHOLD = 2  # Seconds of silence before stopping
    p = pyaudio.PyAudio() 
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    silence_start = None
    start_recording = False

    forgetfactor = 1
    adjustment = 0.02
    threshold = 200


    # Collect initial frames for setting initial level and background
    init_frames = []
    for _ in range(10):
        data = stream.read(CHUNK, exception_on_overflow=False)
        init_frames.append(data)

    # Calculate the average energy of the first 10 frames for initial background
    background_energy_list = [EnergyPerSampleInDecibel(frame) for frame in init_frames]
    background = np.mean(background_energy_list)
    print("background:",background)
    # Set initial level to the energy of the first frame
    level = background_energy_list[0]
    # Start recording on space bar press
    print("Press the space bar to start recording")
    with keyboard.Events() as events:
        for event in events:
            if isinstance(event, keyboard.Events.Press) and event.key == keyboard.Key.space:
                print("Recording started!")
                break

    # Recording loop
    print("Recording... No speech detected will stop the recording.")
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)


        isSpeech, background, level = classifyFrame(data, background, forgetfactor, adjustment, threshold, level)
        if isSpeech:
            frames.append(data)
        print("back:",background, "isSpeech", isSpeech)
        if isSpeech:
            # if silence_start is None:
            #     silence_start = len(frames)
            # elif len(frames) - silence_start >= SILENCE_THRESHOLD * RATE / CHUNK:
            start_recording = True
            time_counter0=time.time()

        if start_recording == True and isSpeech == False:
            frames.append(data)
            time_counter1=time.time()
            if time_counter1-time_counter0>1:
                break
 




        # else:
        #     silence_start = None

        print(f"Current volume: {EnergyPerSampleInDecibel(data)}, Time count: {len(frames) * CHUNK / RATE}")

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data to a wave file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

record_audio()
