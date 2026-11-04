;ksa-lab4 - Karolina Grodz 188664 Kajetan Powala-Mosinski 193054
;kalkulator odwrotnej notacji polskiej (ONP)
;
; - liczby wielocyfrowe 0..255 oddzielane spacja, np.: 12 30 + 4 *
; - operatory: + - * /
; - Enter oblicza wynik
; - wykrywane bledy: za malo argumentow, przekroczenie zakresu 0..255,
;   dzielenie przez zero, przepelnienie stosu, nieznany znak,
;   zbyt wiele liczb na stosie po zakonczeniu wyrazenia

.equ cin,7FH
.equ cout,7DH
.equ newline,95h
.equ header,8000H

.equ wsk,     30H          ; wskaznik wierzcholka stosu kalkulatora
.equ bufor,   40H          ; stos kalkulatora 40H..4FH (16 pozycji)
.equ buf_kon, 50H          ; pierwszy adres ZA buforem

; Rejestry:
; R2 - ostatnio odebrany znak
; R3 - wczytywana liczba,  R7 - flaga "liczba w toku" (0/1)
; R4 - lewy argument,      R5 - prawy argument

.org	header
.db	0A5H,0E5H,0E0H,0A5H
.db	35,255,0,0
.db	0,0,0,0
.db	0,0,0,0
.db	0,0,0,0
.db	0,0,0,0
.db	0,0,0,0
.db	255,255,255,255
.db	"Kalkulator ONP",0
.org	header+64

start:
	mov sp,#4FH            ; stos procesora od 50H w gore - nie koliduje z buforem

petla0:
	mov dptr,#menu
	lcall wypisanie
	mov wsk,#bufor         ; pusty stos kalkulatora
	mov R3,#0
	mov R7,#0

petla:
	lcall cin
	mov R2,A               ; zachowaj znak
	lcall cout             ; echo
	mov A,R2
	cjne A,#0DH,nie_enter
	ljmp oblicz

nie_enter:
	; ---- cyfra? ----
	clr C
	subb A,#30H
	jc nie_cyfra
	mov R6,A               ; wartosc cyfry
	clr C
	subb A,#0AH
	jnc nie_cyfra
	; R3 = R3*10 + cyfra
	mov A,R3
	mov B,#10
	mul AB
	xch A,B                ; A = starszy bajt iloczynu
	jnz cyf_blad           ; wynik > 255
	xch A,B                ; A = mlodszy bajt
	add A,R6
	jc cyf_blad
	mov R3,A
	mov R7,#1
	ljmp petla
cyf_blad:
	ljmp blad_zakres

nie_cyfra:
	lcall zakoncz_liczbe   ; odloz liczbe, jesli byla wczytywana
	jnc nc_ok
	ljmp blad_stos
nc_ok:
	mov A,R2
	cjne A,#20H,nie_spacja ; spacja - separator
	ljmp petla
nie_spacja:
	cjne A,#0AH,op_plus    ; LF - pomin
	ljmp petla

op_plus:
	cjne A,#2BH,op_minus
	lcall pobierz
	jc blad_arg
	mov A,R4
	add A,R5
	jc blad_zakres
	sjmp odloz

op_minus:
	cjne A,#2DH,op_mnoz
	lcall pobierz
	jc blad_arg
	mov A,R4
	clr C
	subb A,R5
	jc blad_zakres         ; wynik ujemny
	sjmp odloz

op_mnoz:
	cjne A,#2AH,op_dziel
	lcall pobierz
	jc blad_arg
	mov A,R4
	mov B,R5
	mul AB
	xch A,B                ; A = starszy bajt iloczynu
	jnz blad_zakres        ; wynik > 255
	xch A,B                ; A = mlodszy bajt
	sjmp odloz

op_dziel:
	cjne A,#2FH,blad_znak
	lcall pobierz
	jc blad_arg
	mov A,R5
	jz blad_zero           ; dzielenie przez 0
	mov A,R4
	mov B,R5
	div AB                 ; iloraz w A

odloz:
	lcall push_a
	jc blad_stos
	ljmp petla

; ---------------- obsluga bledow ----------------
blad_arg:
	mov dptr,#txt_arg
	sjmp blad
blad_zakres:
	mov dptr,#txt_zakres
	sjmp blad
blad_zero:
	mov dptr,#txt_zero
	sjmp blad
blad_stos:
	mov dptr,#txt_stos
	sjmp blad
blad_znak:
	mov dptr,#txt_znak
	sjmp blad
blad_nadmiar:
	mov dptr,#txt_nadmiar

blad:
	lcall newline
	lcall wypisanie
	mov A,R2
	cjne A,#0DH,pomin      ; pomin reszte linii az do Entera
	ljmp petla0
pomin:
	lcall cin
	cjne A,#0DH,pomin
	ljmp petla0

; ---------------- Enter: wynik ----------------
oblicz:
	lcall zakoncz_liczbe
	jc blad_stos
	mov A,wsk
	cjne A,#bufor,ob_niepusty
	lcall newline          ; pusta linia - zacznij od nowa
	ljmp petla0
ob_niepusty:
	cjne A,#bufor+1,blad_nadmiar   ; na stosie musi zostac dokladnie 1 liczba
	lcall newline
	lcall pop_a
	lcall wynik
	lcall newline
	ljmp petla0

; ---------------- podprogramy ----------------

; odklada R3 na stos, jesli wczytywano liczbe; C=1 przy przepelnieniu
zakoncz_liczbe:
	clr C
	mov A,R7
	jz zl_kon
	mov A,R3
	lcall push_a
	mov R3,#0
	mov R7,#0
zl_kon:
	ret

; zdejmuje dwa argumenty: R4 - lewy, R5 - prawy; C=1 gdy za malo
pobierz:
	mov A,wsk
	clr C
	subb A,#bufor+2
	jc pob_kon
	lcall pop_a
	mov R5,A
	lcall pop_a
	mov R4,A
	clr C
pob_kon:
	ret

; odklada A na stos kalkulatora; C=1 gdy stos pelny
push_a:
	mov R1,A
	mov A,wsk
	cjne A,#buf_kon,push_ok
	setb C
	ret
push_ok:
	mov R0,A
	mov A,R1
	mov @R0,A
	inc wsk
	clr C
	ret

pop_a:
	dec wsk
	mov R0,wsk
	mov A,@R0
	ret

; wypisuje A (0..255) dziesietnie bez zer wiodacych
wynik:
	mov B,#100
	div AB                 ; A = setki, B = reszta
	mov R6,A
	mov A,B
	mov B,#10
	div AB                 ; A = dziesiatki, B = jednosci
	mov R5,A
	mov R4,B
	mov A,R6
	jz bez_setek
	add A,#30H
	lcall cout
	mov A,R5               ; po setkach dziesiatki zawsze (np. 105)
	sjmp druk_dz
bez_setek:
	mov A,R5
	jz bez_dz
druk_dz:
	add A,#30H
	lcall cout
bez_dz:
	mov A,R4
	add A,#30H
	lcall cout
	ret

wypisanie:
	clr A
	movc A,@A+dptr
	jz koniec
	lcall cout
	inc dptr
	sjmp wypisanie
koniec:
	ret

menu:
.db "Kalkulator ONP: liczby 0-255, + - * /",13,10,0
txt_arg:
.db "Blad: za malo argumentow",13,10,0
txt_zakres:
.db "Blad: wynik poza zakresem 0-255",13,10,0
txt_zero:
.db "Blad: dzielenie przez zero",13,10,0
txt_stos:
.db "Blad: przepelnienie stosu",13,10,0
txt_znak:
.db "Blad: nieznany znak",13,10,0
txt_nadmiar:
.db "Blad: za duzo liczb na stosie",13,10,0
.end
