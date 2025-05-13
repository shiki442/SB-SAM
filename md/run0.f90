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
  integer :: i

  call get_command_line_args
  ! ! Initialize the random number generator
  ! call random_seed()

  ! ! Generate random values for e1 and e2
  ! call random_number(e1)
  ! call random_number(e2)

  ! ! Scale the random values to the desired range
  ! e1 = e1 * 0.1
  ! e2 = e2 * 0.1

  ! Open the output file
  open(unit=50, file=trim(output_dir)//'/norm200.dat')

  ! Initialize the deformation gradient as an identity matrix
  defm = 0d0
  defm(1,1) = 1.0d0
  defm(2,2) = 1.0d0
  defm(3,3) = 1.0d0

  ! Add the random values to the deformation gradient
  defm(1,1) = defm(1,1) + e1
  defm(1,2) = defm(1,2) + e2

  ! Compute the stress at zero and finite temperature
  call cb1(defm, stress1)

  ! Write the data to the output file
  write(50, '(10E20.10)') stress1(1,:)
  write(50, '(10E20.10)') stress1(2,:)
  write(50, '(10E20.10)') stress1(3,:)

  ! Close the output file
  close(50)

end program run0
