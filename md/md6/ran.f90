Module ran_mod
  ! random number generator
  Implicit None

Contains

  Real(kind=8) Function DLARAN (ISEED)  
    !
    !  -- LAPACK auxiliary routine (version 3.0) --
    !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
    !     Courant Institute, Argonne National Lab, and Rice University
    !     February 29, 1992
    !
    !     .. Array Arguments ..
    Integer :: ISEED (4)  
    !     ..
    !
    !  Purpose
    !  =======
    !
    !  DLARAN returns a random real number from a uniform (0,1)
    !  distribution.
    !
    !  Arguments
    !  =========
    !
    !  ISEED   (input/output) INTEGER array, dimension (4)
    !          On entry, the seed of the random number generator; the array
    !          elements must be between 0 and 4095, and ISEED(4) must be
    !          odd.
    !          On exit, the seed is updated.
    !
    !  Further Details
    !  ===============
    !
    !  This routine uses a multiplicative congruential method with modulus
    !  2**48 and multiplier 33952834046453 (see G.S.Fishman,
    !  'Multiplicative congruential random number generators with modulus
    !  2**b: an exhaustive analysis for b = 32 and a partial analysis for
    !  b = 48', Math. Comp. 189, pp 331-344, 1990).
    !
    !  48-bit integers are stored in 4 integer array elements with 12 bits
    !  per element. Hence the routine is portable across machines with
    !  integers of 32 bits or more.
    !
    !  =====================================================================
    !
    !     .. Parameters ..
    Integer :: M1, M2, M3, M4  
    Parameter (M1 = 494, M2 = 322, M3 = 2508, M4 = 2549)  
    DOUBLEPRECISION ONE  
    Parameter (ONE = 1.0D+0)  
    Integer :: IPW2  
    DOUBLEPRECISION R  
    Parameter (IPW2 = 4096, R = ONE / IPW2)  
    !     ..
    !     .. Local Scalars ..
    Integer :: IT1, IT2, IT3, IT4  
    !     ..
    !     .. Intrinsic Functions ..
    Intrinsic DBLE, MOD  
    !     ..
    !     .. Executable Statements ..
    !
    !     multiply the seed by the multiplier modulo 2**48
    !
    IT4 = ISEED (4) * M4  
    IT3 = IT4 / IPW2  
    IT4 = IT4 - IPW2 * IT3  
    IT3 = IT3 + ISEED (3) * M4 + ISEED (4) * M3  
    IT2 = IT3 / IPW2  
    IT3 = IT3 - IPW2 * IT2  
    IT2 = IT2 + ISEED (2) * M4 + ISEED (3) * M3 + ISEED (4) * M2  
    IT1 = IT2 / IPW2  
    IT2 = IT2 - IPW2 * IT1  
    IT1 = IT1 + ISEED (1) * M4 + ISEED (2) * M3 + ISEED (3) * M2 + &
         ISEED (4) * M1
    IT1 = Mod (IT1, IPW2)  
    !
    !     return updated seed
    !
    ISEED (1) = IT1  
    ISEED (2) = IT2  
    ISEED (3) = IT3  
    ISEED (4) = IT4  
    !
    !     convert 48-bit integer to a real number in the interval (0,1)
    !
    DLARAN = R * (Dble (IT1) + R * (Dble (IT2) + R * (Dble (IT3) &
         + R * (Dble (IT4) ) ) ) )
    Return  
    !
    !     End of DLARAN
    !
  End Function DLARAN


  Real(kind=8) Function DLARND (IDIST, ISEED)  
    !
    !  -- LAPACK auxiliary routine (version 3.0) --
    !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
    !     Courant Institute, Argonne National Lab, and Rice University
    !     September 30, 1994
    !
    !     .. Scalar Arguments ..
    Integer :: IDIST  
    !     ..
    !     .. Array Arguments ..
    Integer :: ISEED (4)  
    !     ..
    !
    !  Purpose
    !  =======
    !
    !  DLARND returns a random real number from a uniform or normal
    !  distribution.
    !
    !  Arguments
    !  =========
    !
    !  IDIST   (input) INTEGER
    !          Specifies the distribution of the random numbers:
    !          = 1:  uniform (0,1)
    !          = 2:  uniform (-1,1)
    !          = 3:  normal (0,1)
    !
    !  ISEED   (input/output) INTEGER array, dimension (4)
    !          On entry, the seed of the random number generator; the array
    !          elements must be between 0 and 4095, and ISEED(4) must be
    !          odd.
    !          On exit, the seed is updated.
    !
    !  Further Details
    !  ===============
    !
    !  This routine calls the auxiliary routine DLARAN to generate a random
    !  real number from a uniform (0,1) distribution. The Box-Muller method
    !  is used to transform numbers from a uniform to a normal distribution.
    !
    !  =====================================================================
    !
    !     .. Parameters ..
    DOUBLEPRECISION ONE, TWO  
    Parameter (ONE = 1.0D+0, TWO = 2.0D+0)  
    DOUBLEPRECISION TWOPI  
    Parameter (TWOPI = 6.2831853071795864769252867663D+0)  
    !     ..
    !     .. Local Scalars ..
    DOUBLEPRECISION T1, T2  
    !     ..
    !     .. External Functions ..
    !DOUBLEPRECISION DLARAN  
    !External DLARAN  
    !     ..
    !     .. Intrinsic Functions ..
    Intrinsic DCOS, DLOG, DSQRT  
    !     ..
    !     .. Executable Statements ..
    !
    !     Generate a real random number from a uniform (0,1) distribution
    !
    T1 = DLARAN (ISEED)  
    !
    If (IDIST.Eq.1) Then  
       !
       !        uniform (0,1)
       !
       DLARND = T1  
    Elseif (IDIST.Eq.2) Then  
       !
       !        uniform (-1,1)
       !
       DLARND = TWO * T1 - ONE  
    Elseif (IDIST.Eq.3) Then  
       !
       !        normal (0,1)
       !
       T2 = DLARAN (ISEED)  
       DLARND = DSqrt ( - TWO * DLog (T1) ) * DCos (TWOPI * T2)  
    Endif
    Return  
    !
    !     End of DLARND
    !
  End Function DLARND

  Function ZLARND(ISEED)
    !
    !  -- LAPACK auxiliary routine (version 3.0) --
    !     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
    !     Courant Institute, Argonne National Lab, and Rice University
    !     September 30, 1994
    !
    !     .. Scalar Arguments ..
    Implicit None
    Real(8) :: Zlarnd(2)
    !     ..
    !     .. Array Arguments ..
    Integer :: ISEED(4)
    !     ..
    !
    !  Purpose
    !  =======
    !
    !  ZLARND returns a random complex number from a uniform or normal
    !  distribution.
    !
    !  Arguments
    !  =========
    !
    !  IDIST   (input) INTEGER
    !          Specifies the distribution of the random numbers:
    !          = 1:  real and imaginary parts each uniform (0,1)
    !          = 2:  real and imaginary parts each uniform (-1,1)
    !          = 3:  real and imaginary parts each normal (0,1)
    !          = 4:  uniformly distributed on the disc abs(z) <= 1
    !          = 5:  uniformly distributed on the circle abs(z) = 1
    !
    !  ISEED   (input/output) INTEGER array, dimension (4)
    !          On entry, the seed of the random number generator; the array
    !          elements must be between 0 and 4095, and ISEED(4) must be
    !          odd.
    !          On exit, the seed is updated.
    !
    !  Further Details
    !  ===============
    !
    !  This routine calls the auxiliary routine DLARAN to generate a random
    !  real number from a uniform (0,1) distribution. The Box-Muller method
    !  is used to transform numbers from a uniform to a normal distribution.
    !
    !  =====================================================================
    !
    !     .. Parameters ..
    Real(8), Parameter  :: ZERO = 0.0D+0, ONE = 1.0D+0, TWO = 2.0D+0 
    Real(8), Parameter  :: TWOPI = 6.2831853071795864769252867663D+0 
    !     ..
    !     .. Local Scalars ..
    Real(8) ::  T1, T2
    !     ..
    !     .. External Functions ..
    !DOUBLE PRECISION   DLARAN
    !EXTERNAL           DLARAN
    !     ..
    !     .. Intrinsic Functions ..
    Intrinsic          DLOG, DSQRT
    !     ..
    !     .. Executable Statements ..
    !
    !     Generate a pair of real random numbers from a uniform (0,1)
    !     distribution
    !
    T1 = DLARAN( ISEED )
    T2 = DLARAN( ISEED )

    ZLARND(1) = DSQRT( -TWO*DLOG( T1 ) )*Dcos(TWOPI*T2 ) 
    Zlarnd(2) = Dsqrt( -Two*Dlog( t1 ) )*Dsin(Twopi*t2 )

  END FUNCTION ZLARND


End Module ran_mod
