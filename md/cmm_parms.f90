MODULE CMM_PARMS

  IMPLICIT NONE

  PUBLIC
  
  ! -- SYSTEM SIZE
  INTEGER, PARAMETER :: NX=12, NY=12, NZ=12, NA=2, NMAX=NX*NY*NZ*NA
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
  integer :: nsamp = 10000   ! 2000

  REAL(8) :: KT, Temperature, Temperature0
  REAL(8) :: time_begin, time_end, time_run
  character(len=100) :: output_dir

  real(8) :: e1, e2


contains

  subroutine set_temp(tmp)
    implicit none
    real(8) :: tmp
    Temperature = tmp
    Temperature0 = tmp
    KT = tmp / EV_D_KB
  end subroutine set_temp

  subroutine get_command_line_args()
    implicit none
    character(len=100) :: arg1, arg2, arg3, arg4, arg5
    real(8) :: tmp
    logical :: dir_exists

    call get_command_argument(1, arg1)
    call get_command_argument(2, arg2)
    call get_command_argument(3, arg3)
    call get_command_argument(4, arg4)
    call get_command_argument(5, arg5)

    output_dir = trim(adjustl(arg1))
    read(arg2, *) tmp
    read(arg3, *) nsamp
    read(arg4, *) e1
    read(arg5, *) e2

    ! inquire(directory=trim(output_dir), exist=dir_exists)
    ! if (.not. dir_exists) then
    !   call system("mkdir -p" // trim(output_dir))
    ! end if

    call set_temp(tmp)

  end subroutine get_command_line_args

  subroutine save_params()
    implicit none

    open(60, file=trim(output_dir)//'/data_params.dat')
    write(60, *) 'num_atoms,', nmax
    write(60, *) 'nx,', nx
    write(60, *) 'ny,', ny
    write(60, *) 'nz,', nz
    write(60, *) 'na,', na
    write(60, *) 'nf,', nf
    write(60, *) 'ndim,', ndim
    write(60, *) 'a0,', a0
    write(60, *) 'KT,', KT
    write(60, *) 'Temperature,', Temperature
    write(60, *) 'nsamp,', nsamp

    close(60)
  end subroutine

END MODULE CMM_PARMS
