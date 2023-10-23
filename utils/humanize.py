def human_readable_number(n):
    # Define the suffixes for powers of 1000
    suffixes = ["", " thousand ", " million ", " billion ", " trillion "]
    # Convert the number to a string and reverse it
    s = str(n)[::-1]
    # Split the string into groups of 3 digits
    groups = [s[i:i+3][::-1] for i in range(0, len(s), 3)]
    # Format each group with commas and suffixes
    result = ""
    for i, g in enumerate(groups):
        if g != "000":
            result = f"{int(g):,}{suffixes[i]}{result}"
            
    return result
