#ifndef CAN_ROUTINE_H
#define CAN_ROUTINE_H

#include "main.h"


extern int  duty_lookup_inlet[3][3];
extern int duty_lookup_outlet[3][3];

extern TaskHandle_t handle_CAN_Subroutine;

extern volatile int  CAN_frame_count;
extern volatile int  CAN_starv_count;

void CAN_Subroutine( void * pvParameters);
void CAN_recover_from_starvation(void);

#endif