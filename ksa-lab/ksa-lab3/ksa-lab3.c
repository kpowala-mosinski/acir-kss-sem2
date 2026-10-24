/*
 * adc_stepper_program.
 *      Zadanie 1 (na bazie demo_program):
 *      - cykliczny odczyt przetwornika ADC,
 *      - wyświetlanie wyniku na 4-cyfrowym wyświetlaczu 7-segmentowym,
 *      - sterowanie silnikiem krokowym z prędkością zależną od wartości z ADC.
 *      Dodatkowo: linijka 8 LED pokazuje poziom ADC, przełącznik DIP nr 0 (IDS0)
 *      wybiera kierunek obrotów.
 *
 *      Compile it with make adc_stepper
 *      (lub: gcc -Wall -O2 -o adc_stepper adc_stepper.c -lwiringPi)
 *      Ctrl+C kończy program (cewki silnika są wyłączane, wyświetlacz gaszony).
 */

#include <stdio.h>
#include <signal.h>
#include <wiringPi.h>

//**** for wiringPI pin enumaration       ****
#define wpA0 5
#define wpA1 6
#define wpA2 25
#define wpAG 2

/*********** BROADCOM port enumeration ***********/
/* UWAGA: numery A0, A1, A2, AG oraz DIK0..DIK7 poprawione względem demo
 * zgodnie ze schematem "RPI hat/MAB interface board" (str. 21):
 * A0..A2, AG = GPIO24..27, DIK0..DIK7 = GPIO16..23.                      */
//output from RPi
#define nRD 2
#define nWR 3
#define PWM0 4
#define PWM1 5
#define A0 24
#define A1 25
#define A2 26
#define AG 27
#define DOK0 8
#define DOK1 9
#define DOK2 10
#define DOK3 11
#define DOK4 12
#define DOK5 13
#define DOK6 14
#define DOK7 15
//intput to RPi
#define INT0 6
#define INT1 7
#define DIK0 16
#define DIK1 17
#define DIK2 18
#define DIK3 19
#define DIK4 20
#define DIK5 21
#define DIK6 22
#define DIK7 23

int OUT_pins [] =                                       //out from RPI
{
       DOK0, DOK1, DOK2, DOK3, DOK4, DOK5, DOK6, DOK7,  //DOK0...DOK7
       A0, A1, A2, AG,                                  //A0...A2, AG
       nRD, nWR,                                        //~RD, ~WR
       PWM0, PWM1                                       //PWM0, PWM1
};

int IN_pins [] =                                        //in to RPI
{
       DIK0, DIK1, DIK2, DIK3, DIK4, DIK5, DIK6, DIK7,  //DIK0...DIK7
       INT0, INT1                                       //int0, int1
};
/*********** BROADCOM port enumeration - end ***********/

/* tablice w numeracji wiringPi – tylko informacyjnie (program używa BCM) */
int wpOUT_pins [] =                                     //out from RPI
{
       10,13,12,14,26,23,15,16,                         //DOK0...DOK7
       5,6,25,2,                                        //A0...A2, AG
       8,9,                                             //~RD, ~WR
       7,21                                             //PWM0, PWM1
};

int wpIN_pins [] =                                      //in to RPI
{
       27,0,1,24,28,29,3,4,                             //DIK0...DIK7
       22,11                                            //int0, int1
};

/*             MAB input definitions  (zapis z RPi do MAB)
addr   BIT7    BIT6      BIT5    BIT4    BIT3    BIT2    BIT1    BIT0
0      OPh4    OPh3      OPh2    OPh1    OKB7    OKB6    OKB5    OKB4
1      OW15    OW14      OW13    OW12    OW11    OW10    OW9     OW8
2      OW7     OW6       OW5     OW4     OW3     OW2     OW1     OW0
3      ODAC7   ODAC6     ODAC5   ODAC4   ODAC3   ODAC2   ODAC1   ODAC0
4      n.c.    n.c.      n.c.    n.c.    n.c.    n.c.    BUZZ    ADCstart
5      LD7     LD6       LD5     LD4     LD3     LD2     LD1     LD0
OPhx - steper motor control,
OKBx - keyboard contacts scan
OWx - 7-segment LCD display control (4xBCD)
ODACx - DAC input data
BUZZ - buzzer control, ADCstart - ADC conversion initialization
LDx - LED line control

               MAB output definitions  (odczyt z MAB do RPi)
addr      BIT7      BIT6    BIT5    BIT4    BIT3    BIT2    BIT1    BIT0
0         IKB3      IKB2    IKB1    IKB0    ----    ----    ----    ----
1         IDS7      IDS6    IDS5    IDS4    IDS3    IDS2    IDS1    IDS0
2         IADC7     IADC6   IADC5   IADC4   IADC3   IADC2   IADC1   IADC0
3         ----      ----    ----    ----    n.c     n.c.    IP1     IP0
4         ----      ----    ----    ----    IVCOout IKOMP   IACstat IK0
5         n.c.      n.c.    n.c.    n.c.    ----    ----    ----    ----
IKBx - keyboard contacts state
IDSx - DPST switches state
IADCx - ADC output data
IPx - SPST switch position
IVCOout - output ov VCO oscillator
IKOMP - comparator output
IACstat - ADC status
IK0 - stepper motor initial position
*/

#define OPh1      0x10        
#define OPh2      0x20
#define OPh3      0x40
#define OPh4      0x80
#define OKB_OFF   0x0F        
#define ADCstart  0x01        
#define BUZZ      0x02
#define IDS0      0x01        
#define IACstat   0x02        
#define IK0       0x01

#define BUS_DELAY_US      2      
#define ADC_TIMEOUT_US    5000   
#define ADC_READY_LEVEL   0      
#define ADC_PERIOD_US     50000  
#define ADC_DEADZONE      8      
#define SPS_MIN           10     
#define SPS_MAX           250    
#define DISPLAY_MILLIVOLTS 0     
#define BLANK             0x0F   

int OPh_seq [] =
{
       OPh1 | OPh4,
       OPh1 | OPh3,
       OPh3 | OPh2,
       OPh2 | OPh4
};

int addr4 = ADCstart;
volatile sig_atomic_t running = 1;

void stopProgram(int sig) { (void)sig; running = 0; }

//****************************************
void setMABAddr(int addr) //BCM enumeration
{
       digitalWrite(AG,1);
       if ( (addr & 1) != 0 ) { digitalWrite(A0,1); } else { digitalWrite(A0,0); } ;
       if ( (addr & 2) != 0 ) { digitalWrite(A1,1); } else { digitalWrite(A1,0); } ;
       if ( (addr & 4) != 0 ) { digitalWrite(A2,1); } else { digitalWrite(A2,0); } ;
       digitalWrite(AG,0);
}

void writeMAB(int data) //BCM enumeration
{
       int pin;
       for (pin = DOK0; pin <= DOK7; ++pin){
               if ( (data & 1) != 0 ) {
                         digitalWrite(pin,1);
               } else {
                         digitalWrite(pin,0);
               } ;
       data = data >> 1;
       }
}

int readMAB( ) //BCM enumeration
{
       int pin, val = 0;
       for (pin = DIK7; pin >= DIK0; --pin){
               val = val << 1;
               val += digitalRead(pin);
       }
       return val;
}

void outMAB(int addr, int data)
{
       setMABAddr(addr);
       writeMAB(data);
       delayMicroseconds(BUS_DELAY_US);
       digitalWrite (nWR, 0) ;
       delayMicroseconds(BUS_DELAY_US);
       digitalWrite (nWR, 1) ;
       delayMicroseconds(BUS_DELAY_US);
}

int inMAB(int addr)
{
       int val;
       setMABAddr(addr);
       digitalWrite (nRD, 0) ;
       delayMicroseconds(BUS_DELAY_US);
       val = readMAB();
       digitalWrite (nRD, 1) ;
       return val;
}

int readADC(void)
{
       unsigned int t0;

       outMAB(4, addr4 & ~ADCstart);
       delayMicroseconds(20);
       outMAB(4, addr4 | ADCstart);
       delayMicroseconds(20);

       t0 = micros();
       while ( ((inMAB(4) & IACstat) ? 1 : 0) != ADC_READY_LEVEL ) {
               if ((unsigned int)(micros() - t0) > ADC_TIMEOUT_US) break;
               delayMicroseconds(20);
       }
       return inMAB(2);
}

void writeDisplay(int value)
{
       int OW[4], i;
       for (i = 0; i < 4; ++i) {
               OW[i] = value % 10;
               value = value / 10;
       }
       for (i = 3; i > 0 && OW[i] == 0; --i) OW[i] = BLANK;
       outMAB(1, (OW[3] << 4) | OW[2]);
       outMAB(2, (OW[1] << 4) | OW[0]);
}

void blankDisplay(void)
{
       outMAB(1, 0xFF);
       outMAB(2, 0xFF);
}

void writeLD(int adc)
{
       int n = (adc * 8 + 127) / 255;
       outMAB(5, (1 << n) - 1);
}

void motorOff(void) { outMAB(0, OKB_OFF); }

unsigned int stepPeriod(int adc)
{
       unsigned int sps;
       if (adc < ADC_DEADZONE) return 0;
       sps = SPS_MIN + (unsigned int)(SPS_MAX - SPS_MIN) * (unsigned int)(adc - ADC_DEADZONE)
                       / (unsigned int)(255 - ADC_DEADZONE);
       return 1000000u / sps;
}

int main (void)
{
     int OUT_pins_nbr = sizeof(OUT_pins) / sizeof(OUT_pins[0]);
     int IN_pins_nbr = sizeof(IN_pins) / sizeof(IN_pins[0]);
     int pin ;
     int adc = 0, adcAcc = 0, phase = 0, cw = 1;
     unsigned int period = 0, newPeriod, now, lastStep, lastADC;


     if (wiringPiSetupGpio() < 0) { //Broadcom enumeration
          fprintf (stderr, "wiringPi setup error (try sudo)\n") ;
          return 1 ;
     }
     signal (SIGINT, stopProgram) ;
     signal (SIGTERM, stopProgram) ;

     digitalWrite (nRD, 1) ;
     digitalWrite (nWR, 1) ;
     digitalWrite (AG, 1) ;
     digitalWrite (PWM0, 0) ;
     digitalWrite (PWM1, 0) ;
     for (pin = 0 ; pin < OUT_pins_nbr ; ++pin) pinMode (OUT_pins[pin], OUTPUT) ;
     for (pin = 0 ; pin < IN_pins_nbr ; ++pin) pinMode (IN_pins[pin], INPUT) ;

     motorOff() ;
     outMAB(4, addr4) ;
     outMAB(5, 0) ;
     blankDisplay() ;

     lastStep = lastADC = micros() - ADC_PERIOD_US ;

     while (running) {
          now = micros() ;

          if (period != 0 && (unsigned int)(now - lastStep) >= period) {
               lastStep += period ;
               if ((unsigned int)(now - lastStep) >= period) lastStep = now ;
               phase = cw ? (phase + 1) & 3 : (phase + 3) & 3 ;
               outMAB(0, OPh_seq[phase] | OKB_OFF) ;
          }

          if ((unsigned int)(now - lastADC) >= ADC_PERIOD_US) {
               lastADC = now ;
               adcAcc += readADC() - (adcAcc >> 2) ;
               adc = (adcAcc + 2) >> 2 ;

#if DISPLAY_MILLIVOLTS
               writeDisplay((adc * 5000 + 127) / 255) ;
#else
               writeDisplay(adc) ;
#endif
               writeLD(adc) ;
               cw = inMAB(1) & IDS0 ;

               newPeriod = stepPeriod(adc) ;
               if (newPeriod == 0 && period != 0) motorOff() ;
               if (newPeriod != 0 && period == 0) lastStep = micros() ;
               period = newPeriod ;

               printf ("\rADC = %3d (%4d mV)  speed = %3u steps/s  dir = %s   ",
                       adc, (adc * 5000 + 127) / 255,
                       period ? 1000000u / period : 0u, cw ? "CW " : "CCW") ;
               fflush (stdout) ;
          }

          delayMicroseconds(200) ;
     }

     motorOff() ;
     blankDisplay() ;
     outMAB(5, 0) ;
     outMAB(4, ADCstart) ;
     digitalWrite (AG, 1) ;
     return 0 ;
}
