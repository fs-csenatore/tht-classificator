from sshkeyboard import listen_keyboard
from dataclasses import dataclass

@dataclass
class Keyboard:
    f5: bool
    f6: bool
    f7: bool

def press(key):
    if key=='f5':
        print('HIT')


listen_keyboard(
    on_press=press,
    delay_second_char=2,
    delay_other_chars=0.1,
)

print('test')