program run0

  !-- Computing the stress from a full atomistic simulation ---
  !-- Given a deformation gradient, we run a molecular      ---
  !-- dynamics simulation, and computed the average stress  ---
  !-- cb0: Cauchy Born at zero temperature;                 ---
  !-- cb1: Cauchy Born at finite temperature.               ---

  use cmm_parms
  use md_stress

  implicit none

  real(8) :: defm(3,3), stress1(3,3)
  real(8) :: e1, e2
  integer :: i, ierr

  call get_command_line_args()

  ! -- deformation gradient
  defm = 0d0
  defm(1,1) = 1.0d0
  defm(2,2) = 1.0d0
  defm(3,3) = 1.0d0

  e1 = 0.01
  e2 = 0.0
  
  defm(1,1) = defm(1,1) + e1
  defm(1,2) = defm(1,2) + e2

  ! -- compute the stress at zero and finite temperature

  open(unit=50, file=trim(output_dir)//'/norm200.dat')

  call cb1(defm, stress1)

  !   
  ! -- data to standard output
  !
  write(50, '(10E20.10)') stress1(1,:)
  write(50, '(10E20.10)') stress1(2,:)
  write(50, '(10E20.10)') stress1(3,:)
 
  close(50)
 
end program run0
