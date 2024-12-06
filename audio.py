from queue import SimpleQueue, Empty

from sdl2.audio import (SDL_GetNumAudioDevices, SDL_AudioSpec, SDL_OpenAudioDevice, SDL_CloseAudioDevice,
    SDL_PauseAudioDevice, SDL_AudioCallback, SDL_AUDIO_ALLOW_ANY_CHANGE, SDL_LockAudioDevice, 
    SDL_GetAudioDeviceName)
from sdl2._sdl_init import SDL_Init, SDL_INIT_AUDIO, SDL_Quit
from sdl2.stdinc import SDL_TRUE, SDL_FALSE


class CallbackPair():
    def __init__(self, recording_device_name=b'USB3. 0 capture Analogue Stereo'):
        self.recording_device_name = recording_device_name
        self.buffer = SimpleQueue()
        self.recording_device_id = None
        self.playback_device_id = None
        
        # If the AudioSpecs are local variables created inside the start method then something seg faults
        # (without printing any extra info). I assume this happens because SDL is trying to access the 
        # AudioSpec objects after they have been destroyed by Python. If they are attached to the CallbackPair, 
        # as below, then this script works. 
        
        self.recording_spec = SDL_AudioSpec(0, 0, 0, 0)
        self.recording_spec.callback = SDL_AudioCallback(self.recording_callback)

        self.playback_spec = SDL_AudioSpec(0, 0, 0, 0)
        self.playback_spec.callback = SDL_AudioCallback(self.playback_callback)

    def recording_callback(self, unused, stream, num_bytes):
        for i in range(num_bytes):
            self.buffer.put(stream[i])

    def playback_callback(self, unused, stream, num_bytes):
        for i in range(num_bytes):
            try:
                stream[i] = self.buffer.get(block=False)
            except Empty:
                stream[i] = 0x00

    def start(self):
        SDL_Init(SDL_INIT_AUDIO)

        requested_recording_spec = SDL_AudioSpec(0, 0, 0, 0)
        requested_recording_spec.callback = SDL_AudioCallback(self.recording_callback)

        requested_playback_spec = SDL_AudioSpec(0, 0, 0, 0)
        requested_playback_spec.callback = SDL_AudioCallback(self.playback_callback) 

        actual_recording_spec = SDL_AudioSpec(0, 0, 0, 0)
        actual_playback_spec = SDL_AudioSpec(0, 0, 0, 0)

        self.recording_device_id = SDL_OpenAudioDevice(self.recording_device_name, SDL_TRUE, 
                                                       self.recording_spec, None, 0)

        self.playback_device_id = SDL_OpenAudioDevice(None, SDL_FALSE, self.playback_spec, None, 0)

        SDL_PauseAudioDevice(self.recording_device_id, SDL_FALSE)
        SDL_PauseAudioDevice(self.playback_device_id, SDL_FALSE)

    def attempt_to_stop(self):
        SDL_LockAudioDevice(self.recording_device_id)
        SDL_LockAudioDevice(self.playback_device_id)
        
        SDL_CloseAudioDevice(self.playback_device_id)

        # At the time of writing (July 2024) SDL may hang when trying to close the recording device.
        print('Trying to close recording device')
        SDL_CloseAudioDevice(self.recording_device_id)
        print('Closed recording device')

        SDL_Quit(SDL_INIT_AUDIO)
        

def list_audio_devices():
    SDL_Init(SDL_INIT_AUDIO)

    for i in range(SDL_GetNumAudioDevices(SDL_TRUE)):
        print(i, SDL_GetAudioDeviceName(i, SDL_TRUE))

    for i in range(SDL_GetNumAudioDevices(SDL_FALSE)):
        print(i, SDL_GetAudioDeviceName(i, SDL_FALSE))

    SDL_Quit(SDL_INIT_AUDIO)    


if __name__ == "__main__":
    from time import sleep
    # from capture_card import display
    # from multiprocessing import Process

    # # Doesn't work as an audio device unless we're getting frames from it
    # cap_card_proc = Process(target=display, daemon=True)
    # cap_card_proc.start()

    # sleep(3)
    
    callback_pair = CallbackPair()
    callback_pair.start()
    input('press key')
    callback_pair.attempt_to_stop()
