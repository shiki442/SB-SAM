module eam_fe

  implicit none

  private

  ! -- parameters in the model
  real(8), parameter :: a0= 2.8665d0
  real(8), parameter :: rcut= 5.67337d0

  real(8), parameter :: A= 5.472212938d0
  real(8), parameter :: E1= 1.749351585d4
  real(8), parameter :: E2= 0.45482d-2 
  real(8), parameter :: F0= -2.1958d0
  real(8), parameter :: F2= 0.67116d0

  real(8), parameter :: h= 0.59906d0
  real(8), parameter :: r00= 0.501724455666748d0
  real(8), parameter :: r01= 1.16319d0
  real(8), parameter :: r02= 4.70161d0
  real(8), parameter :: r03=-1.80420d2
  real(8), parameter :: r04=-6.48409d2
  real(8), parameter :: alpha1= 4.50082d0
  real(8), parameter :: alpha2= 2.23721d0
  real(8), parameter :: beta1 = 0.57200d-2
  real(8), parameter :: beta2 = 8.58106d2
  real(8), parameter :: delta =-0.02924d0

  real(8),dimension(4),parameter::q=(/-0.46026d0,-0.10846d0,-0.93056d0,0.577085d0/)
  real(8),dimension(5),parameter::s=(/0.5d0, -1.5d0, 0.5d0, 5d0,-10d0/)
  real(8),dimension(5),parameter::rho= (/1.1d0,1.2d0,1.6d0,2.0d0,2.5d0/)

  public :: rcut, pair_pot, e_den, eam_en

contains

  subroutine pair_pot(r, p, dp, d2p)

    implicit none

    real(8), intent(in) :: r
    real(8), intent(out):: p, dp, d2p
    real(8) :: x, v1, dv1, ddv1, v2, dv2, ddv2, ps, dps, ddps


    if(r.ge.rcut) then
       p= 0d0; dp=0d0; d2p=0d0
    else
       call morse(r, r01, alpha1, V1, dV1, ddV1)
       call morse(r, r02, alpha2, V2, dV2, ddV2)

       x= (r-rcut)/h
       call psi(x, ps, dps, ddps)
       dps = dps/h
       ddps= ddps/h/h

       p  = (E1*V1+E2*V2+delta)*ps
       dp = (E1*dV1+E2*dV2)*ps + (E1*V1+E2*V2+delta)*dps
       d2p= (E1*ddV1+E2*ddV2)*ps&
            + 2d0*(E1*dV1+E2*dV2)*dps + (E1*V1+E2*V2+delta)*ddps

    end if

    return

  end subroutine pair_pot

  subroutine e_den(r, p, dp, d2p)
    
    implicit none
    
    real(8), intent(in) :: r
    real(8), intent(out):: p, dp, d2p

    real(8) :: x, v1, dv1, ddv1, v2, dv2, ddv2, ps, dps, ddps

    if(r.ge.rcut) then
       p= 0d0; dp=0d0; d2p=0d0
    else
       V1  = dexp(-beta1*( (r-r03)*(r-r03)-(r00-r03)*(r00-r03) ) )
       dV1 = V1*(-beta1*2d0*(r-r03))
       ddV1= V1*(-2d0*beta1+(beta1*2d0*(r-r03))**2)

       V2  = dexp(-beta2*(r-r04))
       dV2 = V2*(-beta2)
       ddV2= V2*beta2*beta2

       x= (r-rcut)/h
       call psi(x, ps, dps, ddps)
       dps = dps/h
       ddps= ddps/h/h

       p  = (A*V1+V2)*ps
       dp = (A*dV1+dV2)*ps + (A*V1+V2)*dps
       d2p= (A*ddV1+ddV2)*ps + 2d0*(A*dV1+dV2)*dps + (A*V1+V2)*ddps

    end if

    return

  end subroutine e_den

  subroutine eam_en(r, p, dp, d2p)
    
    implicit none
    
    real(8), intent(in) :: r
    real(8), intent(out):: p, dp, d2p
    real(8) :: v, dv, d2v, d3v, rr, r2, r3
    integer :: k


    if(r.le.rho(1)) then
       call fp(r, p, dp, d2p, d3v)          
    else
       call fp(r, v, dv, d2v, d3v)
       
       rr= r - rho(1); r2= rr*rr; r3= r2*rr;
       
       p  = v + dv*rr + d2v*r2/2d0 + d3v*r3/6d0
       dp = dv + d2v*rr + d3v*r2/2d0
       d2p= d2v + d3v*rr
       
       do k=1, 5
          call rhorhok(r, k, v, dv, d2v)
          p  = p  + v
          dp = dp + dv
          d2p= d2p+ d2v
       end do

    end  if

    return

  end subroutine eam_en


  subroutine morse(r, r0, alpha, V, dV, d2V)

    ! -- the Morse potential

    implicit none
    
    real(8), intent(in)  :: r, r0, alpha
    real(8), intent(out) :: V, dV, d2V
    real(8) :: tm1, tm2
    
    tm1= dexp(-2d0*alpha*(r-r0))
    tm2= dexp(-alpha*(r-r0))

    V  = tm1 - 2d0*tm2
    
    dV = tm1 - tm2
    dV = dV*(-2d0*alpha)
    
    d2V= tm1*2d0 - tm2
    d2V= d2V*2d0*alpha*alpha

    return

  end subroutine morse

  subroutine psi(x, p, dp, d2p)

    ! -- the smoothing function psi(x)

    implicit none
    
    real(8) :: x
    real(8) :: p, dp, d2p

    if(x.ge.0d0) then
       p=0d0; dp=0d0; d2p=0d0; return

    else
       p  = 1d0 - 1d0/(1d0 + x**4)
       dp = 4d0*x**3/(1d0+x**4)**2
       d2p= (12d0*x**2 - 20d0*x**6)/(1d0+x**4)**3
    end if
       
    return
  end subroutine psi


  subroutine rhorhok(r, k, v, dv, d2v)

    ! -- the additional terms s(k)*(rho-rho(k))**4 

    implicit none
    
    integer, intent(in) :: k
    real(8), intent(in) :: r
    real(8), intent(out):: v, dv, d2v
    real(8) :: rr, r2, r3, r4

    if(r.le.rho(k)) then
       v=0d0; dv=0d0; d2v=0d0; return 
    else
       rr= r - rho(k); r2= rr*rr; r3= r2*rr; r4=r3*rr
       v  = s(k)*r4
       dv = s(k)*4d0*r3
       d2v= s(k)*12d0*r2
    end if

    return

  end subroutine rhorhok

  subroutine fp(r, p, dp, d2p, d3p)
    
    ! -- the function Fp(rho)

    implicit none
    
    real(8), intent(in) :: r
    real(8), intent(out):: p, dp, d2p, d3p
    real(8) :: t, t2, t3, t4, t5, t6 
    
    t= r - 1d0
    t2= t*t; t3= t2*t; t4=t3*t; t5=t4*t; t6=t5*t
    
    ! the third term
    p   = F0 + .5d0*F2*t2+q(1)*t3+q(2)*t4+q(3)*t5+q(4)*t6
    dp  = F2*t+q(1)*3*t2+q(2)*4*t3+q(3)*5*t4+q(4)*6*t5
    d2p = F2  +q(1)*6*t+q(2)*12*t2+q(3)*20*t3+q(4)*30*t4
    d3p = q(1)*6+q(2)*24*t+q(3)*60*t2+q(4)*120*t3
 
    return

  end subroutine fp
    
end module eam_fe
