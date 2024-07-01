from threading import Thread
from time import sleep

def call_at_interval(period, callback, args):
    while(True):
        sleep(period)
        callback(*args)

def set_interval(period, callback, *args):
    Thread(target=call_at_interval, args=(period, callback, args)).start()

def print_a_thing(str_to_print):
    print(str_to_print)

set_interval(1, print_a_thing, "hey its a thread!")
