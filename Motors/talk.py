from gtts import gTTS
from pygame import mixer
from tempfile import TemporaryFile


def say_mama():
    tts = gTTS(text='Mamaaaaa a', lang='es')
    mixer.init()

    sf = TemporaryFile()
    tts.write_to_fp(sf)
    sf.seek(0)

    mixer.music.load(sf)
    mixer.music.play()
