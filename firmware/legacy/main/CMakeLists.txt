idf_component_register(SRCS "can_routine.c" "main.c" "../include/mcp2515.c"
                    INCLUDE_DIRS "." "../include"
                    REQUIRES esp_adc
                    REQUIRES driver
                    REQUIRES nvs_flash
                    REQUIRES app_update
                    )

if(NOT DEFINED VALVE_TYPE)
    set(VALVE_TYPE 1)
endif()

if(VALVE_TYPE EQUAL 0)
    set(VNEMA_FIRMWARE_VARIANT "7mm")
else()
    set(VNEMA_FIRMWARE_VARIANT "DT")
endif()

target_compile_definitions(${COMPONENT_LIB} PRIVATE
    VALVE_TYPE=${VALVE_TYPE}
    VNEMA_FIRMWARE_VERSION=\"${PROJECT_VER}\"
    VNEMA_FIRMWARE_VARIANT=\"${VNEMA_FIRMWARE_VARIANT}\"
)
message(STATUS "VNEMA VALVE_TYPE=${VALVE_TYPE} (1=DT/big valve, 0=7mm valve), version=${PROJECT_VER}")