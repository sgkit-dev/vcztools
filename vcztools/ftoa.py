

def ftoa(f, precision=3):
    buf = ""

    if f < 0:
        buf += '-'
        f = -f

    i = round(f * 10**precision)
    buf += str(i // 10**precision)

    d = [0] * precision

    for idx in range(precision):
        d[idx] = (i // 10**idx) % 10

    buf += "."
    for idx in range(precision):
        sum_ = 0
        for j in d:
            sum_ += j
        if sum_ > 0:
            buf += str(d.pop())

    if buf[-1] == ".":
        buf += "0"

    return buf
