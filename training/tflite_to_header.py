def convert_to_c_array(tflite_file, header_file):
    with open(tflite_file, "rb") as f:
        data = f.read()

    with open(header_file, "w") as f:
        f.write('#include <Arduino.h>\n\n')
        f.write('extern "C" {\n')
        f.write('const unsigned char trained_model[] PROGMEM = {\n  ')
        for i, byte in enumerate(data):
            if i % 12 == 0 and i != 0:
                f.write("\n  ")
            f.write("0x{:02x}, ".format(byte))
        f.write("\n};\n")
        f.write("const unsigned int trained_model_len = {};\n".format(len(data)))
        f.write('}\n')

convert_to_c_array("trained_model.tflite", "trained_model.h")