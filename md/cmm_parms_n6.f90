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

  ! -- number of equilibration steps
  integer, parameter :: nequi= 50000   ! 2000


  REAL(8) :: KT, Temperature
  REAL(8) :: time_begin, time_end, time_run
  character(len=100) :: output_dir
contains

  subroutine set_temp(tmp)
    implicit none
    real(8) :: tmp
    Temperature = tmp
    KT = tmp / EV_D_KB
  end subroutine set_temp

  subroutine get_command_line_args()
    implicit none
    character(len=100) :: arg1, arg2
    real(8) :: tmp

    call get_command_argument(1, arg1)
    call get_command_argument(2, arg2)
    read(arg1, *) tmp
    output_dir = trim(adjustl(arg2))

    call set_temp(tmp)
  end subroutine get_command_line_args

  subroutine save_params()
    open(50, file=trim(output_dir)//'/data_params.txt', position='append')

    write(50, *) 'num_atoms,', nmax
    write(50, *) 'nx,', nx
    write(50, *) 'ny,', ny
    write(50, *) 'nz,', nz
    write(50, *) 'na,', na
    write(50, *) 'nf,', nf
    write(50, *) 'ndim,', ndim
    write(50, *) 'a0,', a0
    write(50, *) 'KT,', KT
    write(50, *) 'Temperature,', Temperature

    close(50)
  end subroutine

END MODULE CMM_PARMS
