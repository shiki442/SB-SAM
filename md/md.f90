module md_stress

  ! ***************************************************************************
  ! **** COMPUTING THE STRESS DIRECTLY FROM THE MOLECULAR DYNAMICS MODELS  ****
  ! ****                                                                   ****


  USE CMM_PARMS
  USE EAM_FE
  USE NOSE_HOOVER
  USE CELL  

  implicit none

  PRIVATE

  ! - atomic degrees of freedom
  real(8), dimension(nmax, 3) :: x, x0, v, f, u
  real(8) :: pot(nmax), rho(nmax), du(nmax)

  ! - Verlet cell list
  real(8), parameter :: rcut2= rcut*rcut
  integer :: list(nmax)
  real(8) :: kv(3,3), axis(3,3) ! -- reciprocal basis, and deformed axis
  real(8) :: bsize(3), csize(3) ! -- boxsize and cell size

  ! - the deformation gradient
  real(8) :: dgrad(3,3)
  ! - parameter for hardystress
  real(8) :: rc = 12d0*2.86449264163d0
  real(8) :: xsamp(3)
  real(8),parameter :: pi=3.1415926
  real(8) :: q(3),p

  PUBLIC CB0, CB1

contains  


  subroutine cb0(defm, stress)

    !:: zero tempearture Cauchy Born 
    !:: stress is computed directly from molecular statics
    !:: since the initial deformation is uniform, no need to run dynamics 
    !:: the reference configuration is the FCC lattice

    implicit none

    real(8), intent(in)  :: defm(3,3)
    real(8), intent(out) :: stress(3,3)
    real(8) :: sigv1(3,3), sigv2(3,3)

    dgrad =  defm

    !:: initialization
    call init_pos
    call comp_force
    call comp_virial_stress(sigv1,sigv2)
    
    stress= sigv1

    return

  end subroutine cb0


  subroutine cb1(defm, stress)

    !:: finite tempearture Cauchy Born 
    !:: stress is computed directly from MD
    !:: the temperature is specified in the cmm_parms.f90
    !:: the reference configuration is the FCC lattice

    implicit none

    real(8), intent(in)  :: defm(3,3)
    real(8), intent(out) :: stress(3,3)

    ! -- total number of steps; number of steps for equilibration; 
    ! -- number of samples in time
    integer, parameter :: nstep= 1000   ! 1000
    integer, parameter :: nequi= 2000   ! 2000
    integer, parameter :: nsf=10, nsamp= (nequi)/nsf
    
    ! -- sampled stress
    real(8) :: sigv1(3,3),sigv2(3,3),hist(nsamp,3,3)
    integer :: ntime, n

    dgrad =  defm
    stress = 0d0

    !:: initialization
    call init_pos
    call save_init_pos
    call init_vel
    call INIT_NHC

    open(10,file='total_energy.dat')
    open(11,file='stress.dat')

    ntime= 0
    do n=1, nstep
       ! -- symmetric operator splitting
       call INTG_NHC
       call INTG_NVE
       call INTG_NHC
       call comp_tp
       write(*,*),n,temperature
       write(10,'(6E16.8)') temperature
    !   if(mod(n,10).eq.0) then
    !        call save_data
    !   end if 
   enddo       

    do n=1,nequi

       call INTG_NHC
       call INTG_NVE
       call INTG_NHC
       call comp_tp
       write(*,*),n,temperature
       write(10,'(6E16.8)') temperature

       if(mod(n,nsf).eq.0) then 
        !  call comp_force
        !  call comp_virial_stress(sigV1,sigv2)
        !  ntime=ntime+1
        !  hist(ntime,:,:)= sigV1+sigv2
        !  hist(ntime,:,:)= sigV1
        !  write(11,'(6E16.8)') sigv1(1,1), hist(ntime,1,2), hist(ntime,1,1)     
          call save_data
       end if

    end do
    !stress= sum(hist,1)/dble(nsamp)

    open(15, file='../data/data_params.txt')

    write(15, *) 'num_atoms,', nmax
    write(15, *) 'nx,', nx
    write(15, *) 'ny,', ny
    write(15, *) 'nz,', nz
    write(15, *) 'na,', na
    write(15, *) 'nf,', nf
    write(15, *) 'ndim,', ndim
    write(15, *) 'a0,', a0
    write(15, *) 'nsample,', nsamp
    write(15, *) 'KT,', KT
    write(15, *) 'Temperature,', Temperature
    write(15, *) 'defm,'
    do n=1,3
       write(15, '(3F10.5)') defm(n, :)
    end do

    return
    close(10);close(11);close(15)

  end subroutine cb1

  subroutine comp_virial_stress(sigma1,sigma2)

    ! -- compute the virial stress
    ! --
    ! -- we are assuming that the verlet list is updated 
    ! -- we also assume du has been computed
    ! -- it is a good idea to call this subroutine right after comp_force
    ! --

    implicit none

    real(8), intent(out):: sigma1(3,3),sigma2(3,3)

    real(8) :: r, r2, dr(3), unit_dr(3), df(3)
    real(8) :: zi(3), zj(3), zij(3), tij(3,3)
    real(8) :: pp, ff, f2, gg, fij(3), vol
    real(8) :: w(nmax,3),ww(3,3),va(3),bb

    integer :: i, j, k, m, j0
    integer :: n, al, be

    sigma1=0d0; sigma2=0d0; w=0d0;ww=0d0;va=0d0

    do m=1, ncell

       i = head(m)
       do while (i .ne. 0) 

          !-- all the particles in the current cell

          j = list(i)
          do while (j .ne. 0) 

             call rij(x(i,:), x(j,:), dr); r2= dot_product(dr,dr)

             ! -------  COMPUTING THE VIRIAL STRESS -------------------
             if (r2 .le. rcut2) then

                r= dsqrt(r2)
                unit_dr= dr/r

                call pair_pot(r, pp, ff, gg); f2= ff
                call e_den(r, pp, ff, gg)

                f2 = f2 + ff*( du(i)+du(j) )
                fij=-f2*unit_dr

                ! -- PBC in the reference coordinate
                zi = x0(i,:); zj= x0(j,:)
                zij= zi - zj
                do k=1, 3
                   zij(k)= zij(k) -Dnint(zij(k)/bsize(k))*bsize(k)
                end do
                zj=zi-zij
                
                bb=bij(zi,zj,xsamp)


                do al=1, 3
                   do be=1, 3
                      ! tij(al, be)= fij(al)*dr(be)
                       tij(al, be)= fij(al)*zij(be)*bb
                   end do
                end do

                sigma1 = sigma1 - tij

             end if
             ! ---------------------------------------------------------

             j = list(j)

          end do

          !--- Part II: particles in neighboring cells
          j0= nnbcell * (m - 1)
          do n=1, nnbcell

             k= map(j0 + n)

             j= head(k)
             do while (j .ne. 0) 

                call rij(x(i,:), x(j,:), dr); r2= dot_product(dr,dr)

                ! -------  COMPUTING THE VIRIAL STRESS -------------------
                if (r2 .le. rcut2) then

                   r= dsqrt(r2)
                   unit_dr= dr/r

                   call pair_pot(r, pp, ff, gg); f2= ff
                   call e_den(r, pp, ff, gg)

                   f2 = f2 + ff*( du(i)+du(j) )
                   fij=-f2*unit_dr

                   ! -- PBC in the reference coordinate
                   zi = x0(i,:); zj= x0(j,:)
                   zij= zi - zj
                   do k=1, 3
                      zij(k)= zij(k) -Dnint(zij(k)/bsize(k))*bsize(k)
                   end do

                   zj=zi-zij
                   bb=bij(zi,zj,xsamp)

                   do al=1, 3
                     do be=1, 3
                     !  tij(al, be)= fij(al)*dr(be)
                      tij(al, be)= fij(al)*zij(be)*bb
                      end do
                   end do

                   sigma1 = sigma1 - tij

                end if
                ! ---------------------------------------------------------

                j = list(j)
             end do

          end do

          i = list(i)
       end do

    end do

  va=sum(v,1)/nmax


do i=1,nmax
   do j=1,3
     w(i,j)=v(i,j)-va(j)
   enddo
enddo



do i=1,nmax
    do al=1,3
      do be=1,3
        tij(al,be)=w(i,al)*w(i,be)
      enddo
    enddo
   sigma2=sigma2-tij
enddo

sigma2=sigma2/vol
    return
  end subroutine comp_virial_stress




  subroutine INTG_NVE

    implicit none

    x= x + v*dt/2d0
    call comp_force
    v= v + f*dt
    x= x + v*dt/2d0

  end subroutine INTG_NVE

SUBROUTINE INIT_NHC

    IMPLICIT NONE

    REAL(8) :: OMEGA2

    ! -- INITIALIZE THE NOSE-HOOVER CHAIN

    GKT= KT; GNKT= Nf*GKT

    OMEGA2 = 20D0 ! WILD GUESS

    ! -- wj WHEN NYS=5
    WNH    = 1D0/(4D0 - DEXP(DLOG(4D0)/3D0))
    WNH(3) = 1D0 - 4D0*WNH(1)

    QMASS(1) = GNKT/OMEGA2+1d0
    QMASS(2:NNOS) = GKT/OMEGA2+1d0

    DT2  = DT/2D0
    DT22 = DT*DT/2D0
    WDTI = WNH*DT/DBLE(NRESN)
    WDTI2= WDTI/2D0
    WDTI4= WDTI/4D0
    WDTI8= WDTI/8D0

    RETURN

  END SUBROUTINE INIT_NHC

  SUBROUTINE INTG_NHC

    ! -- TIME INTEGRATION FOR A NOSE-HOOVER CHAIN
    ! -- THIS SUBROUTINE CORRESPONDS TO THE TIME INTEGRATION
    ! -- W.R.T TO THE OPERATOR L(NHC) FOR HALF OF THE TIME STEP

    implicit none

    REAL(8) :: SCALE, AKIN, AA
    INTEGER :: IRESN, IYOSH, INOS

    SCALE= 1d0
    AKIN = sum(v*v)

    GLOGS(1) = (AKIN - GNKT)/QMASS(1)

    ! -- START THE MULTIPLE TIME STEP PROCEDURE

    DO IRESN=1, NRESN
       DO IYOSH=1, NYOSH

          VLOGS(NNOS)= VLOGS(NNOS) + GLOGS(NNOS)*WDTI4(IYOSH)

          DO INOS=1, NNOS-1
             AA = DEXP(-WDTI8(IYOSH)*VLOGS(NNOS1-INOS))

             VLOGS(NNOS-INOS)= VLOGS(NNOS-INOS)*AA*AA &
                  + WDTI4(IYOSH)*GLOGS(NNOS-INOS)*AA

          END DO

          ! -- UPDATE THE VELOCITY OF THE NH ATOMS
          AA = DEXP(-WDTI2(IYOSH)*VLOGS(1))
          SCALE = SCALE * AA

          ! -- UPDATE THE FORCES ON THE NH ATOMS
          GLOGS(1) = (SCALE*SCALE*AKIN - GNKT)/QMASS(1)

          ! -- UPDATE THE PARTICLE POSITIONS 
          XLOGS = XLOGS + VLOGS*WDTI2(IYOSH)

          ! -- UPDATE THE VELOCITY OF THE NH ATOMS
          DO INOS=1, NNOS-1
             AA = DEXP(-WDTI8(IYOSH)*VLOGS(INOS+1))
             VLOGS(INOS) = VLOGS(INOS)*AA*AA&
                  + WDTI4(IYOSH)*GLOGS(INOS)*AA

             GLOGS(INOS+1) = (QMASS(INOS)*VLOGS(INOS)*VLOGS(INOS)&
                  - GKT)/QMASS(INOS+1)
          END DO
          VLOGS(NNOS) = VLOGS(NNOS) + GLOGS(NNOS)*WDTI4(IYOSH)
       END DO
    END DO

    ! -- UPDATE THE VELOCITY OF THE ATOMS
    v = v*SCALE

    RETURN

  END SUBROUTINE INTG_NHC



  subroutine init_pos

    implicit none

    real(8) :: vol2, bv(3,3), shft(3)
    integer :: i, j, k, n, l, m

    bsize= (/dble(nx)*a0, dble(ny)*a0, dble(nz)*a0/)

    x(1, :)= (/0d0,   0d0,   0d0/)
    x(2, :)= (/a0/2d0,  a0/2d0, a0/2d0/)

    n=0

    Do i=1, nx
       Do j=1, ny
          Do k=1, nz
             shft=  (/dble(i-1)*a0, dble(j-1)*a0, dble(k-1)*a0/)
             Do l=1, na
                x(n+l, :)= x(l, :) + shft
             End Do
             n=n+na
          End Do
       End Do
    End Do

    x0= x

    ! -- uniform deformation
    do n=1, nmax
       x(n,:)= matmul(dgrad, x0(n,:))
    end do

    ! ............................................................. ....
    ! ADDED May 22, 2012
    ! we use the Verlet cell list to speed up the force calculation
    ! ...

    bv(1,:)= (/1d0, 0d0, 0d0/)
    bv(2,:)= (/0d0, 1d0, 0d0/)
    bv(3,:)= (/0d0, 0d0, 1d0/)

    ! -- deform the basis vectors accordingly
    do m=1, 3
       bv(m,:)= matmul(dgrad, bv(m,:))
    end do

    ! -- the inverse
    vol2= bv(1,1)*bv(2,2)-bv(1,2)*bv(2,1)
    kv(3,:)= bv(3,:)
    kv(2,:)= (/-bv(1,2), bv(1,1), 0d0/)/vol2
    kv(1,:)= (/ bv(2,2),-bv(2,1), 0d0/)/vol2  

    ! -- the three axes in the deformed state
    axis(1,:)= bsize(1)*bv(1,:)
    axis(2,:)= bsize(2)*bv(2,:)
    axis(3,:)= bsize(3)*bv(3,:)

    ! -- cell size
    csize(1)= bsize(1)/mx
    csize(2)= bsize(2)/my
    csize(3)= bsize(3)/mz

    ! -- initialize the verlet cells
    call init_cell_maps

    ! ................................................................


    ! -- compute the force to get the integrator started
    call comp_force

    return

  end subroutine init_pos

  subroutine init_vel

    use ran_mod

    implicit none

    real(8) :: v0(3)
    integer :: n, k, isd(4)=(/7, 8, 9, 11/)

    do n=1, nmax
       do k=1, 3
          v(n,k) = dlarnd(3,isd)*dsqrt(KT)
       end do
    end do

    ! -- force the average to be zero
    v0= sum(v,1)/dble(nmax)
    do k=1, 3
       v(:,k) = v(:,k) - v0(k)
    end do

    return

  end subroutine init_vel
  

  subroutine comp_tp
    integer :: i,j  
    temperature = 0d0  
    do i = 1,nmax
       do j = 1,3
          temperature = temperature + v(i,j)*v(i,j)/dble(nmax)/3d0/k2ev
       end do
    end do
    return
  end subroutine comp_tp






  subroutine comp_force

    ! -- compute the force

    implicit none

    real(8) :: r, r2, dr(3), unit_dr(3), df(3)
    real(8) :: pp, ff, f1, f2, gg, d2f, sig, eta

    integer :: i, j, j0, k, m, n

    ! -- first move atoms to cells
    call link_atom2cell

    f= 0d0; rho= 0d0;  pot=0d0

    do m=1, ncell

       i = head(m)

       do while (i .ne. 0) 

          !--- Part I: all the particles in the current cell

          j = list(i)
          do while (j .ne. 0) 

             call rij(x(i,:), x(j,:), dr)
             r2= dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

             if (r2 .le. rcut2) then

                r= dsqrt(r2)
                unit_dr= dr/r

                call pair_pot(r, pp, ff, gg)

                df = ff*unit_dr
                f(i, :)= f(i,:) - df
                f(j, :)= f(j,:) + df

                !                pot(i) = pot(i) + pp/2d0
                !                pot(j) = pot(j) + pp/2d0 

                call e_den(r, pp, ff, gg)       
                rho(i) = rho(i) + pp
                rho(j) = rho(j) + pp

             end if

             j=list(j)
          end do

          !--- Part II: particles in neighboring cells
          j0= nnbcell * (m - 1)

          do n=1, nnbcell

             k= map(j0 + n)

             j= head(k)

             do while (j .ne. 0) 

                call rij(x(i,:), x(j,:), dr)
                r2= dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

                if (r2 .le. rcut2) then

                   r= dsqrt(r2)
                   unit_dr= dr/r

                   call pair_pot(r, pp, ff, gg)
                   df = ff*unit_dr

                   f(i, :)= f(i,:) - df
                   f(j, :)= f(j,:) + df

                   !                   pot(i) = pot(i) + pp/2d0
                   !                   pot(j) = pot(j) + pp/2d0 

                   call e_den(r, pp, ff, gg)       
                   rho(i) = rho(i) + pp
                   rho(j) = rho(j) + pp

                end if

                j=list(j)

             end do

          end do

          i= list(i)

       end do
    end do

    ! -- second step, embedded energy
    do i = 1, nmax          
       call eam_en(rho(i), pp, ff, gg)
       du(i)= ff
       pot(i) = pot(i) + pp
    end do

    ! -- third step: force due to embedded energy

    do m=1, ncell

       i = head(m)
       do while (i .ne. 0) 

          !-- all the particles in the current cell

          j = list(i)
          do while (j .ne. 0) 

             call rij(x(i,:), x(j,:), dr)
             r2= dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

             if (r2 .le. rcut2) then

                r= dsqrt(r2)
                unit_dr= dr/r

                call e_den(r, pp, ff, gg)

                f1= ff*du(i)
                f2= ff*du(j)

                df = (f1 + f2)*unit_dr

                f(i, :)= f(i, :) - df
                f(j, :)= f(j, :) + df

             end if

             j = list(j)

          end do

          !--- Part II: particles in neighboring cells
          j0= nnbcell * (m - 1)
          do n=1, nnbcell

             k= map(j0 + n)

             j= head(k)
             do while (j .ne. 0) 

                call rij(x(i,:), x(j,:), dr)
                r2= dr(1)*dr(1)+dr(2)*dr(2)+dr(3)*dr(3)

                if (r2 .le. rcut2) then

                   r= dsqrt(r2)
                   unit_dr= dr/r

                   call e_den(r, pp, ff, gg)

                   f1= ff*du(i)
                   f2= ff*du(j)

                   df = (f1 + f2)*unit_dr

                   f(i, :)= f(i, :) - df
                   f(j, :)= f(j, :) + df

                end if

                j = list(j)
             end do

          end do

          i = list(i)
       end do

    end do

    return

  end subroutine comp_force

  subroutine link_atom2cell

    implicit none

    real(8) :: Rx, Ry, Rz
    integer :: n, m, ii, jj, kk

    ! -- sort out all the atoms ...
    head = 0
    do n=1, nmax

       ! -- the position in the reference coordinate
       Rx = dot_product(x(n,:), kv(1,:))
       Ry = dot_product(x(n,:), kv(2,:))
       Rz = dot_product(x(n,:), kv(3,:))

       ii = int(Rx/csize(1)) + 1
       jj = int(Ry/csize(2)) + 1
       kk = int(Rz/csize(3)) + 1

       m  = icell(ii, jj, kk)

       list(n)= head(m)
       head(m)= n
    end do

    return

  end subroutine link_atom2cell

  subroutine rij(ri, rj, dr)

    ! -- apply the periodic boundary condition around a deformed state

    implicit none
    real(8), intent(in) :: ri(3), rj(3)
    real(8), intent(out):: dr(3)

    real(8) :: dr0(3), trm
    integer :: m

    dr0= ri - rj; dr= dr0
    do m=1, 3
       trm= dot_product(dr0, kv(m,:))/bsize(m)
       dr = dr - Dnint(trm)*axis(m,:)
    end do

    return

  end subroutine rij

 subroutine save_data

    implicit none

    integer, save :: mt=0
    character :: num*5

    integer :: i, j, k, m, n, n1, n2    

    num= num2str(mt)
    open(12, file='../data/pos-f.dat', position='append')

    write(12, '(6E20.10)') (x(i, :), f(i, :), i = 1, nmax)

    close(12)

   !  open(12, file='../data/pos-f'//num//'.dat')
   !  open(13, file='../data/pos-f'//num//'.xyz')

   !  write(13, *) nmax
   !  write(13, *) 'Molecular dynamics of Aluminum under PBC'
   !  do i=1, nmax
       
   !     write(12, '(6E20.10)') x(i, :), f(i,:)
   !     write(13, '(A5,6E20.10)') 'Al', x(i, :), f(i,:)

   !  end do
   !  close(12); close(13)

    mt= mt +1

    return

  end subroutine save_data

 subroutine save_init_pos

    implicit none

    integer :: i, j, k, m, n, n1, n2    

    open(14, file='../data/init_pos.dat')

    write(14, '(6E20.10)') (x(i, :), i = 1, nmax)

    close(14)

    return

  end subroutine save_init_pos

  function num2str(num) result(str)
   !  integer, intent(in) :: num
   !  character(len=3) :: str
   !  write(str, '(I3)') num
    integer, intent(in) :: num
    character(len=5) :: str

    str = char(int(num/10000) + 48) // &
      char(mod(num,10000)/1000 + 48) // &
      char(mod(num,1000)/100 + 48) // &
      char(mod(num,100)/10 + 48) // &
      char(mod(num,10) + 48) 
    
    return
  end function num2str

   !:::::::::::::::::::::::: bond function :::::::::::::::::::::::::

  function bij(xi, xj, x) result(b)
    ! -- compute the bond function, bij, using a simpson's rule
    implicit none
    real(8) :: xi(3), xj(3), x(3), b , xjj(3),dr(3)
    integer, parameter :: N=100
    real(8) :: h=1d0/dble(N)
    real(8) :: lam, f(0:N)
    integer :: m
	
	
     do m=0, N
       lam= dble(m)*h       
       f(m)= psi(xi,xj,x,lam)
     end do
     b= 0d0
     do m=0, N-2, 2
       b= b + (f(m)+4d0*f(m+1)+f(m+2))*h/3d0
     end do
     return
  end function bij

  !:::::::::::::::::::: value of phi between x=xi and x=xj ::::::::::::::::::::::::::::::::::

  function psi(xi, xj, x, lambda) result(ps)
    implicit none
    real(8) :: xi(3), xj(3), x(3), lambda, ps
    real(8) :: y(3), xij(3)
    xij = xi - xj
    y = x - ( xj + lambda*xij )
    ps = phi(y)
    return
  end function psi

  !:::::::::::::::::: kernel function :::::::::::::::::::::::::::::::::::::::::::::::

  function phi(x) result(res)
    implicit none
    real(8),intent(in) :: x(3)
    real(8) :: res
    res=psi2(x(1)/rc,x(2)/rc,x(3)/rc)/(rc**3)
  end function phi
  
  
   function psi2(x, y, z) result(ps)
    ! -- regularizing the delta function using a cosine kernel
    implicit none
    real(8) :: x, y, z, ps
    if(dabs(x).le.1d0.and.dabs(y).le.1d0.and.dabs(z).le.1d0) then
       ps = (1d0 + dcos(pi*x))/2d0
       ps = ps*(1d0 + dcos(pi*y))/2d0
       ps = ps*(1d0 + dcos(pi*z))/2d0
    else
       ps= 0d0
    end if
    return
  end function psi2




end module md_stress
