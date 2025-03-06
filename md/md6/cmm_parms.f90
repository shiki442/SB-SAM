MODULE CMM_PARMS

  IMPLICIT NONE

  PUBLIC
  
  ! -- SYSTEM SIZE
  INTEGER, PARAMETER :: NX=6, NY=6, NZ=6, NA=2, NMAX=NX*NY*NZ*NA
  INTEGER, PARAMETER :: NDIM=3

  ! -- TOTAL NUMBER OF DEGREES OF FREEDOM
  INTEGER, PARAMETER :: NF= NMAX*NDIM

  ! -- STEP SIZE
  REAL(8), PARAMETER :: DT= .5E-1

  ! -- SYSTEM TEMPERATURE
  REAL(8), PARAMETER :: EV_D_KB= 1.16045D4
  real(8), parameter :: k2ev = 8.6173324d-5
  ! -- lattice constant for analytical potential
  REAL(8) :: a0= 2.86449264163d0

  ! -- lattice constant for cubic spline potential
  !REAL(8) :: a0= 2.8665000245055d0

  REAL(8) :: KT,Temperature
  REAL(8) :: time_begin, time_end, time_run

contains

  subroutine set_temp(tmp)

    implicit none

    real(8) :: tmp
    
    KT= tmp/EV_D_KB

  end subroutine set_temp

END MODULE CMM_PARMS
