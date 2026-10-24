# Makefile dla zadania 1 (MAB-94 + Raspberry Pi, wiringPi)
CC      ?= gcc
CFLAGS  = -O2 -Wall -I/usr/local/include
LDFLAGS = -L/usr/local/lib
LDLIBS  = -lwiringPi -lpthread -lm -lrt

adc_stepper: adc_stepper.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f adc_stepper
