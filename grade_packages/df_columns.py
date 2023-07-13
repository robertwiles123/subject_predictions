def combined_full_original():
    return ['FFT20', 'SEN need(s)', 'PP', 'Mock 1', 'Mock 2', 'Mock 3']

def combined_full_clean():
    return ['FFT20', 'PP', 'Mock 1', 'Mock 2', 'Mock 3', 'SEN bool']

def combined_bool():
    return ['PP', 'SEN bool']

def combined_independent():
    return ['Mock 3']

def combined_dependent():
    return ['FFT20', 'PP', 'Mock 1', 'Mock 2', 'SEN bool']

def triple_full_original():
    return ["FFT20", "SEN need(s)", "PP", "year 10 bio grade", "year 10 chem grade", "year 10 phys grade", "year 11 paper 1 bio grade", "year 11 paper 1 chem grade", "year 11 paper 1 phys grade", "year 11 paper 2 bio grade", "year 11 paper 2 chem grade", "year 11 paper 2 phys grade"]

def triple_full_clean():
    return ["FFT20", "PP", "year 10 bio grade", "year 10 chem grade", "year 10 phys grade", "year 11 paper 1 bio grade", "year 11 paper 1 chem grade", "year 11 paper 1 phys grade", "year 11 paper 2 bio grade", "year 11 paper 2 chem grade", "year 11 paper 2 phys grade", "SEN bool"]

def triple_non_grades():
    return ['PP', 'SEN bool', 'FFT20']

def triple_bool():
    return ['PP', 'SEN bool']

def triple_grades():
    return ["year 10 bio grade", "year 10 chem grade", "year 10 phys grade", "year 11 paper 1 bio grade", "year 11 paper 1 chem grade", "year 11 paper 1 phys grade"]

def triple_independent():
    return ["year 11 paper 2 bio grade", "year 11 paper 2 chem grade", "year 11 paper 2 phys grade"]

def triple_dependent():
    return ["FFT20", "PP", "year 10 bio grade", "year 10 chem grade", "year 10 phys grade", "year 11 paper 1 bio grade", "year 11 paper 1 chem grade", "year 11 paper 1 phys grade", "SEN bool"]