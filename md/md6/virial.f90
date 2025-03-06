subroutine comp_virial_stress

  ! -- compute the virial stress
  ! -- this is the average stress over the entire system
  ! -- we assume the verlet list has already been updated from previous steps
  ! -- we also assume du has been previously computed
  ! -- this can be guaranteed by a call of comp_force

  implicit none

  real(8) :: r, r2, dr(3), dr0(3), unit_dr(3), df(3)
  real(8) :: pp, ff, f2, d2f

  real(8) :: vol

  integer :: jbeg, jnb, jend, nlist
  integer :: nn, lst(nnb), nm

  integer :: i, j, k, m, al, be

  vstr=0d0

  do i = 1, nmax-1

     jbeg = point(i)
     jend = point(i+1)-1
     do jnb= jbeg, jend
        j= list(jnb)

        dr = s(i,:) - s(j,:) 
        dr0= x0(i,:) - x0(j,:)
        do k=1, 3
           dr(k) = dr(k) - Dnint(dr(k)/bsize(k))*bsize(k)
           dr0(k)= dr0(k) -Dnint(dr0(k)/bsize(k))*bsize(k)
        end do

        dr= matmul(defm, dr)
        r= Dsqrt(dr(1)*dr(1) + dr(2)*dr(2) + dr(3)*dr(3))

        if (r .le. rcut) then

           unit_dr= dr/r

           call pair_pot(r, pp, ff, d2f)
           f2= ff

           call e_den(r, pp, ff, d2f)
           f2= f2 + ff*(du(i)+du(j))
           df= f2*unit_dr

           do al=1, 3
              do be=1, 3
                 vstr(al,be)=vstr(al,be)+ df(al)*dr0(be)
              end do
           end do

        end if
     end do
  end do

  vol = bsize(1)*bsize(2)*bsize(3)
  vstr= vstr/vol

  return

end subroutine comp_virial_stress
