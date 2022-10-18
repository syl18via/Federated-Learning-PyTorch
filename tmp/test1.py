
from test2 import TARGET, update_fn, print_fn
import test2 as relative

def log(s):
    print("test1: " + s)

log(f"TARGET before updating {TARGET}")
log(f"test2.TARGET before updating {test2.TARGET}")
print_fn()

print("")
log("Update in test2.py")
update_fn(1)
log(f"TARGET after updating {TARGET}")
log(f"test2.TARGET after updating {test2.TARGET}")
print_fn()

print("")
log("Update TARGET")
TARGET = 2
log(f"TARGET after updating {TARGET}")
log(f"test2.TARGET after updating {test2.TARGET}")
print_fn()

print("")
log("Update test2.TARGET")
test2.TARGET = 3
log(f"TARGET after updating {TARGET}")
log(f"test2.TARGET after updating {test2.TARGET}")
print_fn()



