module cell

  ! -- THIS IS A MODULE TO IMPLEMENT THE CELL LIST ALGORITHM --
  ! -- HERE WE DECIDE THE NUMBER OF CELLS, AND ARRANGE THEM  --

  USE CMM_PARMS

  PUBLIC 

  ! -- SIZE OF THE CELLS
  integer, parameter :: mx=4, my=4, mz=4
  integer, parameter :: nnbcell=13
  integer, parameter :: ncell= mx*my*mz, mapsize= nnbcell*ncell
  integer :: head(ncell), map(mapsize)

contains 

  function icell(i, j, k) result(m)

    !:: number the cells

    implicit none

    integer :: i, j, k, m

    m= 1 + mod(i - 1 + mx, mx) &
         + mod(j - 1 + my, my) * mx &
         + mod(k - 1 + mz, mz) * my * mx

    return

  end function icell
    

  subroutine init_cell_maps

    ! set up a list of neighboring cells
    
    implicit none
    
    integer :: i, j, k, imap
    
    do k=1, mz
       do j=1, my
          do i=1, mx
             
             imap= ( icell( i, j, k) -1 )*nnbcell
             
             !:: link to the following cells
             map( imap + 1  ) = icell( i + 1, j    , k     )
             map( imap + 2  ) = icell( i + 1, j + 1, k     )
             map( imap + 3  ) = icell( i    , j + 1, k     )
             map( imap + 4  ) = icell( i - 1, j + 1, k     )
             map( imap + 5  ) = icell( i + 1, j    , k - 1 )
             map( imap + 6  ) = icell( i + 1, j + 1, k - 1 )
             map( imap + 7  ) = icell( i    , j + 1, k - 1 )
             map( imap + 8  ) = icell( i - 1, j + 1, k - 1 )
             map( imap + 9  ) = icell( i + 1, j    , k + 1 )
             map( imap + 10 ) = icell( i + 1, j + 1, k + 1 )
             map( imap + 11 ) = icell( i    , j + 1, k + 1 )
             map( imap + 12 ) = icell( i - 1, j + 1, k + 1 )
             map( imap + 13 ) = icell( i    , j    , k + 1 )
             
          end do
       end do
    end do
    
    return

  end subroutine init_cell_maps

end module cell
