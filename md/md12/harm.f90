module harm_approx

  ! -- Quasi-harmonic approximation of the finite temperature CauchyBorn stress ---
  ! -- 

  use eam_fe
  use cmm_parms

  implicit none

  PRIVATE

  ! -- number of k-points in each direction; k-points; volume of the unit cell
  integer, parameter :: nq= 16
  real(8) :: bz(nq,nq,nq,3), vol

  ! -- the force constants 
  integer, parameter :: nfmax=1600
  integer :: nfc
  real(8), allocatable :: rn(:,:), Dn(:,:,:), DDn(:,:,:,:,:)

  ! -- basis vectors for fcc lattice; reciporacal basis
  real(8), dimension(3) :: t1, t2, t3, k1, k2, k3 
  
  PUBLIC CB2

contains

  subroutine cb2(defm, stress)

    ! -- compute the higher order term in the harmonic approximation

    implicit none

    real(8), intent(in)  :: defm(3,3)
    real(8), intent(out) :: stress(3,3)

    real(8), parameter   :: eps=.5e-6
    real(8), allocatable :: Dn1(:,:,:), Dn2(:,:,:), Dn0(:,:,:)

    real(8) :: defm0(3,3), defm1(3,3), defm2(3,3)

    real(8) :: f(3,3), z(3), g(3,3), d(3,3), h(3,3), t(3,3), r
    integer :: i, j, k, al, be, n

    call init_eam
    call init_basis
    call init_bz

    allocate(rn(nfmax,3), Dn(nfmax,3,3), DDn(nfmax,3,3,3,3) )
    allocate(Dn1(nfmax,3,3), Dn2(nfmax,3,3),Dn0(nfmax,3,3) )

    ! -- compute the derivative of the force constant w.r.t. the deformation gradient
    defm0= defm
    do al=1, 3
       do be=1, 3
          defm1= defm; defm1(al,be) = defm(al, be) - eps; 
          call fconst(defm1); Dn1= Dn
          defm2= defm; defm2(al,be) = defm(al, be) + eps; 
          call fconst(defm2); Dn2= Dn
          DDn(:,:,:,al,be)= (Dn2- Dn1)/(2d0*eps)
       end do
    end do
    call fconst(defm0); Dn0= Dn

    f= 0d0
    do i=1, nq
       do j=1, nq
          do k=1, nq
             z= bz(i,j,k,:)

             r= norm(z)
             if(r.ge.1e-5) then

             !:: compute the dynamical matrix
             Dn= Dn0; call dynm(z, d); G= inv(d); 
             
             do al=1, 3
                do be=1, 3
                   Dn= DDn(:,:,:,al,be)
                   call dynm(z,t) 
                   f(al,be)= f(al, be) + mat_dot_prod(G,t)
                end do
             end do
          
          end if
          end do
       end do
    end do
    f= f/dble(nq*nq*nq)/vol/2d0

    stress = f

    deallocate(rn, Dn, DDn, Dn0, Dn1, Dn2)

    return

  end subroutine cb2

  function mat_dot_prod(A, B) result(d)
    
    ! -- compute the dot product of two matrices 

    implicit none

    real(8) :: A(3,3), B(3,3), d
    integer :: i, j

    d= 0d0
    do i=1, 3
       do j=1, 3
          d= d + A(i,j)*B(i,j)
       end do
    end do

    return

  end function mat_dot_prod

  subroutine init_BZ
    
    ! -- initialize the Brilloiun Zone

    implicit none

    real(8) :: z(3), x(3), s, y(3), t
    integer :: i, j, k, ii, jj, kk

    do i=1, nq
       do j=1, nq
          do k=1, nq
             z  = dble(2*i-nq)/dble(2*nq)*k1    &
                  + dble(2*j-nq)/dble(2*nq)*k2    &
                  + dble(2*k-nq)/dble(2*nq)*k3
             
             !:: find the corresponding point in the B zone
             x= z
             s= norm(x)
             do ii=-1, 1
                do jj=-1, 1
                   do kk=-1, 1
                      y= z + dble(ii)*k1 + dble(jj)*k2 + dble(kk)*k3
                      
                      t= norm(y)
                      
                      if(t.lt.s) then
                         x= y; s= t
                      end if
                   end do
                end do
             end do
             
             !z= x

             bz(i,j,k,:)= z
          end do
       end do
    end do

    return

  end subroutine init_BZ

  subroutine dynm(xi, D)

    ! -- compute the dynamic matrix given the deformation gradient and 
    ! -- the wavenumber \xi

    implicit none

    real(8), intent(in) :: xi(3)
    real(8), intent(out):: D(3,3)

    real(8) :: x(3), A(3,3), trm
    integer :: n

    D= 0d0
    do n=1, nfc

       x= rn(n,:); A= Dn(n,:,:)

       trm= dot_product(x, xi)

       D= D + dcos(trm)*A
    end do


  end subroutine dynm

  subroutine fconst(defm)

    ! -- given the deformation gradient, we compute the force constants
        
    implicit none

    real(8), intent(in) :: defm(3,3)
    integer, save :: init=0

    real(8) :: rho, r, r0, ri, rj, pp, dF, d2F, ff, hh, de, de2, d2p, d2e, dp
    real(8) :: dr(3), dr0(3), dri(3), drj(3), drij(3), h(3,3)
    integer :: i, j, k, n, ii, jj, kk, ncount, mm, nn

    ! -- find atoms within a radius 2*rcut
    ! -- the force constants will be computed for these atoms
    ! -- the force constants for other atoms are zero
    if(init.eq.0) then
       ncount=0
       do i=-6, 6
          do j=-6, 6
             do k=-6, 6
                dr= dble(i)*t1+dble(j)*t2+dble(k)*t3
                r= norm(dr)
                if(r.gt.1d-4.and.r.le.2.5d0*rcut) then
                   ncount= ncount + 1
                   rn(ncount,:)= dr
                end if
             end do
          end do
       end do
       ! -- the last atom is at the origin
       ncount= ncount + 1 
       rn(ncount,:)= 0d0
       nfc= ncount
       init=1
    end if
    
    !:: compute the electron density ...
    rho= 0d0
    do n=1, nfc-1
       dr= rn(n,:)
       dr= matmul(defm, dr)
       r= norm(dr)
       if(r.le.rcut) then
          call density(r, pp, ff, hh)
          rho = rho + pp
       end if
    end do

    !:: compute the derivatives of the embedded energy
    call embd_en(rho, pp, dF, d2F)
    
    ! :: compute the force constant matrix
    do n=1, nfc-1
       dr0= rn(n,:)
       r0 = norm(dr0)
       dr = matmul(defm, dr0)
       r  = norm(dr)
       call pair(r, pp, dp, d2p)
       call density(r, pp, de, d2e)

       do mm=1, 3
          do nn=1, 3   
             h(mm,nn)= ((dp/r - d2p) + 2d0*dF*(de/r-d2e))*dr(mm)*dr(nn)/(r*r) &
                  - (dp+2d0*dF*de)/r*delta(mm,nn)    
          end do
       end do

       do ii=-10, 10
          do jj=-10, 10
             do kk=-10, 10
                dri= dble(ii)*t1+dble(jj)*t2+dble(kk)*t3
                dri= matmul(defm, dri)
                         
                drj= dri - dr
                ri = norm(dri)  
                rj = norm(drj)
                         
                if(ri .gt. 1D-4 .and. ri.le. rcut &
                     .and. rj .gt. 1D-4 .and. rj.le.rcut) then
                            
                   call density(ri, pp, de, d2e); de2= de;
                   call density(rj, pp, de, d2e); de2= de2*de
                   
                   do mm=1, 3
                      do nn=1, 3   
                         h(mm,nn)= h(mm,nn) + d2F*de2*dri(mm)*drj(nn)/ri/rj
                      end do
                   end do
                   
                end if
                         
             end do
          end do
       end do

       Dn(n,:,:)= h
 
    end do
    
    ! -- the force constants add up to zero 
    Dn(nfc,:,:)= -sum(Dn(1:nfc-1,:,:), 1) 

    return


  end subroutine fconst

  subroutine init_basis
    
    implicit none

    real(8) :: pi

    pi= datan2(0d0, -1d0)

    vol= a1**3d0/2D0
    
    t1= a1*(/ .5d0, .5d0,-.5D0/)       !:: basis of the primitive cell 
    t2= a1*(/-.5D0, .5D0, .5d0/)
    t3= a1*(/ .5D0,-.5D0, .5d0/)
    
    k1= 2d0*pi/vol*a1*a1*(/ .5d0, .5d0,  0d0/)
    k2= 2d0*pi/vol*a1*a1*(/  0d0, .5d0, .5d0/)
    k3= 2d0*pi/vol*a1*a1*(/ .5d0,  0d0, .5d0/)

    return

  end subroutine init_basis

  function delta(m, n) result(r)

    implicit none

    integer, intent(in) :: m, n
    real(8) :: r

    r=0d0; if(m.eq.n) r=1d0

  end function delta

  function norm(v) result(s)

    implicit none

    real(8), intent(in) :: v(3)
    real(8) :: s
    
    s= dsqrt(v(1)*v(1) + v(2)*v(2) + v(3)*v(3))

    return

  end function norm

  Function inv(A) Result(B)
    
    !:: the inverse of a 3x3 matrix

    Implicit None
    real(8), intent(in) :: A(3, 3)
    real(8) :: B(3, 3), d

    B(1, 1)= a(2,2)*a(3,3) - a(3,2)*a(2,3)
    B(1, 2)= a(1,3)*a(3,2) - a(1,2)*a(3,3)
    B(1, 3)= a(1,2)*a(2,3) - a(2,2)*a(1,3)
    B(2, 1)= a(2,3)*a(3,1) - a(2,1)*a(3,3)
    B(2, 2)= a(1,1)*a(3,3) - a(1,3)*a(3,1)
    B(2, 3)= a(1,3)*a(2,1) - a(1,1)*a(2,3)
    B(3, 1)= a(2,1)*a(3,2) - a(2,2)*a(3,1)
    B(3, 2)= a(1,2)*a(3,1) - a(1,1)*a(3,2)
    B(3, 3)= a(1,1)*a(2,2) - a(1,2)*a(2,1)

    d= det(A)

    B= B/d

    Return
    
  End Function inv

  Function det(A) Result(d)

    !:: determinant of a 3x3 matrix

    Implicit None
    
    real(8), intent(in) :: A(3,3)
    real(8) :: d
    
    d= a(1,1)*(a(2,2)*a(3,3) - a(2,3)*a(3,2)) &
         - a(1,2)*(a(2,1)*a(3,3) - a(2,3)*a(3,1)) &
         + a(1,3)*(a(2,1)*a(3,2) - a(2,2)*a(3,1))
    
    Return
    
  End Function det

end module harm_approx
