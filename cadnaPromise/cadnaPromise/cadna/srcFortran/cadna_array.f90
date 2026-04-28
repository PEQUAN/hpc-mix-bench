
! Module de surcharge des fonctions tableaux






module cadna_array
  use cadna_types
  use cadna_affect
  use cadna_add
  use cadna_mult
  use cadna_math
  use cadna_ne

  implicit none




  interface random_number
    module procedure random_number_single_st_1D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_1D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_2D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_2D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_3D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_3D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_4D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_4D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_5D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_5D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_6D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_6D
  end interface random_number
  interface random_number
    module procedure random_number_single_st_7D
  end interface random_number
  interface random_number
    module procedure random_number_double_st_7D
  end interface random_number



  interface dot_product
    module procedure dot_product_single_st
  end interface dot_product
  interface dot_product
    module procedure dot_product_double_st
  end interface dot_product



  interface matmul
    module procedure matmul_single_st_1D2D
  end interface matmul
  interface matmul
    module procedure matmul_double_st_1D2D
  end interface matmul
  interface matmul
    module procedure matmul_single_st_2D1D
  end interface matmul
  interface matmul
    module procedure matmul_double_st_2D1D
  end interface matmul
  interface matmul
    module procedure matmul_single_st_2D2D
  end interface matmul
  interface matmul
    module procedure matmul_double_st_2D2D
  end interface matmul



  interface sum
    module procedure sum_single_st_1D
  end interface sum
  interface sum
    module procedure sum_double_st_1D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_1D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_1D
  end interface sum
  interface sum
    module procedure sum_single_st_2D
  end interface sum
  interface sum
    module procedure sum_double_st_2D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_2D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_2D
  end interface sum
  interface sum
    module procedure sum_single_st_3D
  end interface sum
  interface sum
    module procedure sum_double_st_3D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_3D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_3D
  end interface sum
  interface sum
    module procedure sum_single_st_4D
  end interface sum
  interface sum
    module procedure sum_double_st_4D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_4D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_4D
  end interface sum
  interface sum
    module procedure sum_single_st_5D
  end interface sum
  interface sum
    module procedure sum_double_st_5D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_5D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_5D
  end interface sum
  interface sum
    module procedure sum_single_st_6D
  end interface sum
  interface sum
    module procedure sum_double_st_6D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_6D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_6D
  end interface sum
  interface sum
    module procedure sum_single_st_7D
  end interface sum
  interface sum
    module procedure sum_double_st_7D
  end interface sum
  interface sum
    module procedure sum_single_st_dim_7D
  end interface sum
  interface sum
    module procedure sum_double_st_dim_7D
  end interface sum

  interface product
    module procedure product_single_st_1D
  end interface product
  interface product
    module procedure product_double_st_1D
  end interface product
  interface product
    module procedure product_single_st_dim_1D
  end interface product
  interface product
    module procedure product_double_st_dim_1D
  end interface product
  interface product
    module procedure product_single_st_2D
  end interface product
  interface product
    module procedure product_double_st_2D
  end interface product
  interface product
    module procedure product_single_st_dim_2D
  end interface product
  interface product
    module procedure product_double_st_dim_2D
  end interface product
  interface product
    module procedure product_single_st_3D
  end interface product
  interface product
    module procedure product_double_st_3D
  end interface product
  interface product
    module procedure product_single_st_dim_3D
  end interface product
  interface product
    module procedure product_double_st_dim_3D
  end interface product
  interface product
    module procedure product_single_st_4D
  end interface product
  interface product
    module procedure product_double_st_4D
  end interface product
  interface product
    module procedure product_single_st_dim_4D
  end interface product
  interface product
    module procedure product_double_st_dim_4D
  end interface product
  interface product
    module procedure product_single_st_5D
  end interface product
  interface product
    module procedure product_double_st_5D
  end interface product
  interface product
    module procedure product_single_st_dim_5D
  end interface product
  interface product
    module procedure product_double_st_dim_5D
  end interface product
  interface product
    module procedure product_single_st_6D
  end interface product
  interface product
    module procedure product_double_st_6D
  end interface product
  interface product
    module procedure product_single_st_dim_6D
  end interface product
  interface product
    module procedure product_double_st_dim_6D
  end interface product
  interface product
    module procedure product_single_st_7D
  end interface product
  interface product
    module procedure product_double_st_7D
  end interface product
  interface product
    module procedure product_single_st_dim_7D
  end interface product
  interface product
    module procedure product_double_st_dim_7D
  end interface product

  interface minval
    module procedure minval_single_st_1D
  end interface minval
  interface minval
    module procedure minval_double_st_1D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_1D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_1D
  end interface minval
  interface minval
    module procedure minval_single_st_2D
  end interface minval
  interface minval
    module procedure minval_double_st_2D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_2D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_2D
  end interface minval
  interface minval
    module procedure minval_single_st_3D
  end interface minval
  interface minval
    module procedure minval_double_st_3D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_3D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_3D
  end interface minval
  interface minval
    module procedure minval_single_st_4D
  end interface minval
  interface minval
    module procedure minval_double_st_4D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_4D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_4D
  end interface minval
  interface minval
    module procedure minval_single_st_5D
  end interface minval
  interface minval
    module procedure minval_double_st_5D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_5D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_5D
  end interface minval
  interface minval
    module procedure minval_single_st_6D
  end interface minval
  interface minval
    module procedure minval_double_st_6D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_6D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_6D
  end interface minval
  interface minval
    module procedure minval_single_st_7D
  end interface minval
  interface minval
    module procedure minval_double_st_7D
  end interface minval
  interface minval
    module procedure minval_single_st_dim_7D
  end interface minval
  interface minval
    module procedure minval_double_st_dim_7D
  end interface minval

  interface maxval
    module procedure maxval_single_st_1D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_1D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_1D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_1D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_2D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_2D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_2D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_2D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_3D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_3D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_3D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_3D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_4D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_4D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_4D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_4D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_5D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_5D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_5D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_5D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_6D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_6D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_6D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_6D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_7D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_7D
  end interface maxval
  interface maxval
    module procedure maxval_single_st_dim_7D
  end interface maxval
  interface maxval
    module procedure maxval_double_st_dim_7D
  end interface maxval

  interface minloc
    module procedure minloc_single_st_1D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_1D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_1D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_1D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_2D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_2D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_2D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_2D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_3D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_3D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_3D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_3D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_4D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_4D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_4D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_4D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_5D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_5D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_5D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_5D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_6D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_6D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_6D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_6D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_7D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_7D
  end interface minloc
  interface minloc
    module procedure minloc_single_st_dim_7D
  end interface minloc
  interface minloc
    module procedure minloc_double_st_dim_7D
  end interface minloc

  interface maxloc
    module procedure maxloc_single_st_1D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_1D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_1D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_1D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_2D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_2D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_2D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_2D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_3D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_3D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_3D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_3D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_4D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_4D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_4D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_4D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_5D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_5D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_5D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_5D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_6D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_6D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_6D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_6D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_7D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_7D
  end interface maxloc
  interface maxloc
    module procedure maxloc_single_st_dim_7D
  end interface maxloc
  interface maxloc
    module procedure maxloc_double_st_dim_7D
  end interface maxloc

contains



  subroutine random_number_single_st_1D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_1D

  subroutine random_number_double_st_1D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_1D

  subroutine random_number_single_st_2D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_2D

  subroutine random_number_double_st_2D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_2D

  subroutine random_number_single_st_3D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2),size(harvest,3)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_3D

  subroutine random_number_double_st_3D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2),size(harvest,3)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_3D

  subroutine random_number_single_st_4D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2), &
                             size(harvest,3),size(harvest,4)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_4D

  subroutine random_number_double_st_4D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2), &
                             size(harvest,3),size(harvest,4)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_4D

  subroutine random_number_single_st_5D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_5D

  subroutine random_number_double_st_5D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_5D

  subroutine random_number_single_st_6D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5),size(harvest,6)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_6D

  subroutine random_number_double_st_6D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5),size(harvest,6)) :: temp

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_6D

  subroutine random_number_single_st_7D(harvest)
    use ISO_FORTRAN_ENV, only : REAL32
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL32), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5),size(harvest,6), &
                             size(harvest,7)) :: temp 

    call random_number(temp)
    harvest = temp
  end subroutine random_number_single_st_7D

  subroutine random_number_double_st_7D(harvest)
    use ISO_FORTRAN_ENV, only : REAL64 
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(out) :: harvest
    ! Local declaration
    real(REAL64), dimension (size(harvest,1),size(harvest,2),size(harvest,3), &
                             size(harvest,4),size(harvest,5),size(harvest,6), &
                             size(harvest,7)) :: temp 

    call random_number(temp)
    harvest = temp
  end subroutine random_number_double_st_7D




  function dot_product_single_st(vector_a, vector_b)
    ! Dummy arguments declaration
    type(single_st), dimension(:), intent(in) :: vector_a, vector_b
    type(single_st) :: dot_product_single_st

    dot_product_single_st = SUM(vector_a*vector_b)
  end function dot_product_single_st

  function dot_product_double_st(vector_a, vector_b)
    ! Dummy arguments declaration
    type(double_st), dimension(:), intent(in) :: vector_a, vector_b
    type(double_st) :: dot_product_double_st

    dot_product_double_st = SUM(vector_a*vector_b)
  end function dot_product_double_st




  function matmul_single_st_1D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in) :: matrix_a
    type(single_st), dimension (:,:), intent(in) :: matrix_b 
    type(single_st), dimension (size(matrix_b,2)) :: matmul_single_st_1D2D
    ! Local declaration
    integer j

    do j=1,size(matrix_b,2)
      matmul_single_st_1D2D(j) = DOT_PRODUCT(matrix_a, matrix_b(:,j))
    enddo
  end function matmul_single_st_1D2D

  function matmul_double_st_1D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in) :: matrix_a
    type(double_st), dimension (:,:), intent(in) :: matrix_b 
    type(double_st), dimension (size(matrix_b,2)) :: matmul_double_st_1D2D
    ! Local declaration
    integer j

    do j=1,size(matrix_b,2)
      matmul_double_st_1D2D(j) = DOT_PRODUCT(matrix_a, matrix_b(:,j))
    enddo
  end function matmul_double_st_1D2D

  function matmul_single_st_2D1D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in) :: matrix_a 
    type(single_st), dimension (:), intent(in) :: matrix_b
    type(single_st), dimension (size(matrix_a,1)) :: matmul_single_st_2D1D
    ! Local declaration
    integer i

    do i=1,size(matrix_a,1)
      matmul_single_st_2D1D(i) = DOT_PRODUCT(matrix_a(i,:), matrix_b)
    enddo
  end function matmul_single_st_2D1D

  function matmul_double_st_2D1D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in) :: matrix_a 
    type(double_st), dimension (:), intent(in) :: matrix_b
    type(double_st), dimension (size(matrix_a,1)) :: matmul_double_st_2D1D
    ! Local declaration
    integer i

    do i=1,size(matrix_a,1)
      matmul_double_st_2D1D(i) = DOT_PRODUCT(matrix_a(i,:), matrix_b)
    enddo
  end function matmul_double_st_2D1D

  function matmul_single_st_2D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in) :: matrix_a 
    type(single_st), dimension (:,:), intent(in) :: matrix_b 
    type(single_st), dimension (size(matrix_a,1), size(matrix_b,2)) :: matmul_single_st_2D2D 
    ! Local declaration
    integer i, j 

    do i=1,size(matrix_a,1)
      do j=1,size(matrix_b,2)
        matmul_single_st_2D2D(i,j) = DOT_PRODUCT(matrix_a(i,:), matrix_b(:,j))
      enddo
    enddo 
  end function matmul_single_st_2D2D

  function matmul_double_st_2D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in) :: matrix_a 
    type(double_st), dimension (:,:), intent(in) :: matrix_b 
    type(double_st), dimension (size(matrix_a,1), size(matrix_b,2)) :: matmul_double_st_2D2D 
    ! Local declaration
    integer i, j 

    do i=1,size(matrix_a,1)
      do j=1,size(matrix_b,2)
        matmul_double_st_2D2D(i,j) = DOT_PRODUCT(matrix_a(i,:), matrix_b(:,j))
      enddo
    enddo 
  end function matmul_double_st_2D2D


  recursive subroutine calcul_indices(indice_1D, indices_nD, profil)
    integer, intent(inout)               :: indice_1D
    integer, dimension(:), intent(inout) :: indices_nD
    integer, dimension(:), intent(in)    :: profil
    integer n, prod

    n = size(profil)
    prod = product(profil)
    if ( n == 0 ) then
      indices_nD(n+1) = indice_1D + 1
      return
    end if
    indices_nD(n+1) = indice_1D/prod + 1
    indice_1D = mod(indice_1D,prod)
    call calcul_indices(indice_1D, indices_nD, profil(:n-1))
  end subroutine calcul_indices



  function minloc_single_st(array, mask, back)
    type(single_st),   dimension(:), intent(in) :: array
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional :: back
    integer minloc_single_st
    integer i, j
    integer first, last, pas
    integer rang
    type(single_st) val_min, temp

    if (present(mask)) then
      if (count(mask) == 0) then
        minloc_single_st = -1
        return
      end if
    end if

    if (present(back)) then
      if (back) then
        first = size(array); last = 1; pas = -1
      else
        first = 1; last = size(array); pas = 1
      end if
    else
      first = 1; last = size(array); pas = 1
    end if
    if (present(mask)) then
      do i=first,last,pas
        if ( mask(i) ) then
          val_min = array(i)
          exit
        end if
      end do
    else
      val_min = array(first)
      i = first
    end if
    rang = i

    do j=i+pas,last,pas
      if (present(mask)) then
        if( .not.mask(j) ) cycle
      end if
      temp = val_min
      val_min = min(val_min, array(j))
      if (val_min /= temp) rang = j
    end do
    minloc_single_st = rang - 1
  end function minloc_single_st

  function minloc_double_st(array, mask, back)
    type(double_st),   dimension(:), intent(in) :: array
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional :: back
    integer minloc_double_st
    integer i, j
    integer first, last, pas
    integer rang
    type(double_st) val_min, temp

    if (present(mask)) then
      if (count(mask) == 0) then
        minloc_double_st = -1
        return
      end if
    end if

    if (present(back)) then
      if (back) then
        first = size(array); last = 1; pas = -1
      else
        first = 1; last = size(array); pas = 1
      end if
    else
      first = 1; last = size(array); pas = 1
    end if
    if (present(mask)) then
      do i=first,last,pas
        if ( mask(i) ) then
          val_min = array(i)
          exit
        end if
      end do
    else
      val_min = array(first)
      i = first
    end if
    rang = i

    do j=i+pas,last,pas
      if (present(mask)) then
        if( .not.mask(j) ) cycle
      end if
      temp = val_min
      val_min = min(val_min, array(j))
      if (val_min /= temp) rang = j
    end do
    minloc_double_st = rang - 1
  end function minloc_double_st

  function maxloc_single_st(array, mask, back)
    type(single_st),   dimension(:), intent(in) :: array
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional :: back
    integer maxloc_single_st
    integer i, j
    integer first, last, pas
    integer rang
    type(single_st) val_max, temp

    if (present(mask)) then
      if (count(mask) == 0) then
        maxloc_single_st = -1
        return
      end if
    end if

    if (present(back)) then
      if (back) then
        first = size(array); last = 1; pas = -1
      else
        first = 1; last = size(array); pas = 1
      end if
    else
      first = 1; last = size(array); pas = 1
    end if
    if (present(mask)) then
      do i=first,last,pas
        if ( mask(i) ) then
          val_max = array(i)
          exit
        end if
      end do
    else
      val_max = array(first)
      i = first
    end if
    rang = i

    do j=i+pas,last,pas
      if (present(mask)) then
        if( .not.mask(j) ) cycle
      end if
      temp = val_max
      val_max = max(val_max, array(j))
      if (val_max /= temp) rang = j
    end do
    maxloc_single_st = rang - 1
  end function maxloc_single_st

  function maxloc_double_st(array, mask, back)
    type(double_st),   dimension(:), intent(in) :: array
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional :: back
    integer maxloc_double_st
    integer i, j
    integer first, last, pas
    integer rang
    type(double_st) val_max, temp

    if (present(mask)) then
      if (count(mask) == 0) then
        maxloc_double_st = -1
        return
      end if
    end if

    if (present(back)) then
      if (back) then
        first = size(array); last = 1; pas = -1
      else
        first = 1; last = size(array); pas = 1
      end if
    else
      first = 1; last = size(array); pas = 1
    end if
    if (present(mask)) then
      do i=first,last,pas
        if ( mask(i) ) then
          val_max = array(i)
          exit
        end if
      end do
    else
      val_max = array(first)
      i = first
    end if
    rang = i

    do j=i+pas,last,pas
      if (present(mask)) then
        if( .not.mask(j) ) cycle
      end if
      temp = val_max
      val_max = max(val_max, array(j))
      if (val_max /= temp) rang = j
    end do
    maxloc_double_st = rang - 1
  end function maxloc_double_st




  function minloc_single_st_1D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(1) :: minloc_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(1) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_1D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_1D, profil(1:size(profil)-1) )
  end function minloc_single_st_1D

  function minloc_double_st_1D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(1) :: minloc_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(1) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_1D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_1D, profil(1:size(profil)-1) )
  end function minloc_double_st_1D

  function minloc_single_st_2D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(2) :: minloc_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(2) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_2D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_2D, profil(1:size(profil)-1) )
  end function minloc_single_st_2D

  function minloc_double_st_2D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(2) :: minloc_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(2) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_2D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_2D, profil(1:size(profil)-1) )
  end function minloc_double_st_2D

  function minloc_single_st_3D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(3) :: minloc_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(3) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_3D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_3D, profil(1:size(profil)-1) )
  end function minloc_single_st_3D

  function minloc_double_st_3D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(3) :: minloc_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(3) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_3D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_3D, profil(1:size(profil)-1) )
  end function minloc_double_st_3D

  function minloc_single_st_4D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(4) :: minloc_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(4) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_4D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_4D, profil(1:size(profil)-1) )
  end function minloc_single_st_4D

  function minloc_double_st_4D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(4) :: minloc_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(4) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_4D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_4D, profil(1:size(profil)-1) )
  end function minloc_double_st_4D

  function minloc_single_st_5D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(5) :: minloc_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(5) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_5D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_5D, profil(1:size(profil)-1) )
  end function minloc_single_st_5D

  function minloc_double_st_5D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(5) :: minloc_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(5) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_5D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_5D, profil(1:size(profil)-1) )
  end function minloc_double_st_5D

  function minloc_single_st_6D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(6) :: minloc_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(6) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_6D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_6D, profil(1:size(profil)-1) )
  end function minloc_single_st_6D

  function minloc_double_st_6D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(6) :: minloc_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(6) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_6D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_6D, profil(1:size(profil)-1) )
  end function minloc_double_st_6D

  function minloc_single_st_7D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(7) :: minloc_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(7) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_single_st(p_array, back=back)
    else
      rang = minloc_single_st(p_array)
    end if
    if ( rang == -1) then
      minloc_single_st_7D = 0
      return
    end if

    call calcul_indices( rang, minloc_single_st_7D, profil(1:size(profil)-1) )
  end function minloc_single_st_7D

  function minloc_double_st_7D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(7) :: minloc_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(7) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = minloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = minloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = minloc_double_st(p_array, back=back)
    else
      rang = minloc_double_st(p_array)
    end if
    if ( rang == -1) then
      minloc_double_st_7D = 0
      return
    end if

    call calcul_indices( rang, minloc_double_st_7D, profil(1:size(profil)-1) )
  end function minloc_double_st_7D

  function maxloc_single_st_1D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(1) :: maxloc_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(1) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_1D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_1D, profil(1:size(profil)-1) )
  end function maxloc_single_st_1D

  function maxloc_double_st_1D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(1) :: maxloc_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(1) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_1D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_1D, profil(1:size(profil)-1) )
  end function maxloc_double_st_1D

  function maxloc_single_st_2D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(2) :: maxloc_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(2) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_2D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_2D, profil(1:size(profil)-1) )
  end function maxloc_single_st_2D

  function maxloc_double_st_2D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(2) :: maxloc_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(2) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_2D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_2D, profil(1:size(profil)-1) )
  end function maxloc_double_st_2D

  function maxloc_single_st_3D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(3) :: maxloc_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(3) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_3D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_3D, profil(1:size(profil)-1) )
  end function maxloc_single_st_3D

  function maxloc_double_st_3D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(3) :: maxloc_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(3) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_3D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_3D, profil(1:size(profil)-1) )
  end function maxloc_double_st_3D

  function maxloc_single_st_4D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(4) :: maxloc_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(4) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_4D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_4D, profil(1:size(profil)-1) )
  end function maxloc_single_st_4D

  function maxloc_double_st_4D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(4) :: maxloc_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(4) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_4D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_4D, profil(1:size(profil)-1) )
  end function maxloc_double_st_4D

  function maxloc_single_st_5D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(5) :: maxloc_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(5) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_5D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_5D, profil(1:size(profil)-1) )
  end function maxloc_single_st_5D

  function maxloc_double_st_5D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(5) :: maxloc_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(5) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_5D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_5D, profil(1:size(profil)-1) )
  end function maxloc_double_st_5D

  function maxloc_single_st_6D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(6) :: maxloc_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(6) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_6D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_6D, profil(1:size(profil)-1) )
  end function maxloc_single_st_6D

  function maxloc_double_st_6D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(6) :: maxloc_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(6) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_6D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_6D, profil(1:size(profil)-1) )
  end function maxloc_double_st_6D

  function maxloc_single_st_7D(array, mask, back)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(7) :: maxloc_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(7) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_single_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_single_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_single_st(p_array, back=back)
    else
      rang = maxloc_single_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_single_st_7D = 0
      return
    end if

    call calcul_indices( rang, maxloc_single_st_7D, profil(1:size(profil)-1) )
  end function maxloc_single_st_7D

  function maxloc_double_st_7D(array, mask, back)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    logical, optional :: back
    integer, dimension(7) :: maxloc_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    integer, dimension(7) :: profil
    integer rang

    profil = shape(array)
    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask

    if (present(mask) .and. present(back)) then
      rang = maxloc_double_st(p_array, mask=p_mask, back=back)
    else if (present(mask)) then
      rang = maxloc_double_st(p_array, mask=p_mask)
    else if (present(back)) then
      rang = maxloc_double_st(p_array, back=back)
    else
      rang = maxloc_double_st(p_array)
    end if
    if ( rang == -1) then
      maxloc_double_st_7D = 0
      return
    end if

    call calcul_indices( rang, maxloc_double_st_7D, profil(1:size(profil)-1) )
  end function maxloc_double_st_7D




  function sum_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_1D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_1D = sum_single_st_1D + p_array(i)
    end do

  end function sum_single_st_1D

  function sum_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_1D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_1D = sum_double_st_1D + p_array(i)
    end do

  end function sum_double_st_1D

  function sum_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_2D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_2D = sum_single_st_2D + p_array(i)
    end do

  end function sum_single_st_2D

  function sum_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_2D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_2D = sum_double_st_2D + p_array(i)
    end do

  end function sum_double_st_2D

  function sum_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_3D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_3D = sum_single_st_3D + p_array(i)
    end do

  end function sum_single_st_3D

  function sum_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_3D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_3D = sum_double_st_3D + p_array(i)
    end do

  end function sum_double_st_3D

  function sum_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_4D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_4D = sum_single_st_4D + p_array(i)
    end do

  end function sum_single_st_4D

  function sum_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_4D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_4D = sum_double_st_4D + p_array(i)
    end do

  end function sum_double_st_4D

  function sum_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_5D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_5D = sum_single_st_5D + p_array(i)
    end do

  end function sum_single_st_5D

  function sum_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_5D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_5D = sum_double_st_5D + p_array(i)
    end do

  end function sum_double_st_5D

  function sum_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_6D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_6D = sum_single_st_6D + p_array(i)
    end do

  end function sum_single_st_6D

  function sum_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_6D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_6D = sum_double_st_6D + p_array(i)
    end do

  end function sum_double_st_6D

  function sum_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: sum_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_single_st_7D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_single_st_7D = sum_single_st_7D + p_array(i)
    end do

  end function sum_single_st_7D

  function sum_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: sum_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`sum', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`sum', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    sum_double_st_7D = 0
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_double_st_7D = sum_double_st_7D + p_array(i)
    end do

  end function sum_double_st_7D


  function product_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_1D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_1D = product_single_st_1D * p_array(i) 
    end do

  end function product_single_st_1D

  function product_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_1D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_1D = product_double_st_1D * p_array(i) 
    end do

  end function product_double_st_1D

  function product_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_2D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_2D = product_single_st_2D * p_array(i) 
    end do

  end function product_single_st_2D

  function product_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_2D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_2D = product_double_st_2D * p_array(i) 
    end do

  end function product_double_st_2D

  function product_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_3D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_3D = product_single_st_3D * p_array(i) 
    end do

  end function product_single_st_3D

  function product_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_3D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_3D = product_double_st_3D * p_array(i) 
    end do

  end function product_double_st_3D

  function product_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_4D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_4D = product_single_st_4D * p_array(i) 
    end do

  end function product_single_st_4D

  function product_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_4D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_4D = product_double_st_4D * p_array(i) 
    end do

  end function product_double_st_4D

  function product_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_5D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_5D = product_single_st_5D * p_array(i) 
    end do

  end function product_single_st_5D

  function product_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_5D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_5D = product_double_st_5D * p_array(i) 
    end do

  end function product_double_st_5D

  function product_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_6D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_6D = product_single_st_6D * p_array(i) 
    end do

  end function product_single_st_6D

  function product_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_6D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_6D = product_double_st_6D * p_array(i) 
    end do

  end function product_double_st_6D

  function product_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: product_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_single_st_7D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_single_st_7D = product_single_st_7D * p_array(i) 
    end do

  end function product_single_st_7D

  function product_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: product_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`product', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`product', 0, 3))')'
    integer i

    p_array(1:size(array)) => array
    product_double_st_7D = 1
    if (present(mask)) p_mask(1:size(mask)) => mask
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_double_st_7D = product_double_st_7D * p_array(i) 
    end do

  end function product_double_st_7D


  function minval_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_1D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_1D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_1D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_1D = min( minval_single_st_1D, p_array(j) )
    end do

  end function minval_single_st_1D

  function minval_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_1D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_1D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_1D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_1D = min( minval_double_st_1D, p_array(j) )
    end do

  end function minval_double_st_1D

  function minval_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_2D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_2D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_2D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_2D = min( minval_single_st_2D, p_array(j) )
    end do

  end function minval_single_st_2D

  function minval_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_2D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_2D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_2D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_2D = min( minval_double_st_2D, p_array(j) )
    end do

  end function minval_double_st_2D

  function minval_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_3D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_3D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_3D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_3D = min( minval_single_st_3D, p_array(j) )
    end do

  end function minval_single_st_3D

  function minval_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_3D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_3D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_3D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_3D = min( minval_double_st_3D, p_array(j) )
    end do

  end function minval_double_st_3D

  function minval_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_4D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_4D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_4D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_4D = min( minval_single_st_4D, p_array(j) )
    end do

  end function minval_single_st_4D

  function minval_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_4D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_4D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_4D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_4D = min( minval_double_st_4D, p_array(j) )
    end do

  end function minval_double_st_4D

  function minval_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_5D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_5D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_5D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_5D = min( minval_single_st_5D, p_array(j) )
    end do

  end function minval_single_st_5D

  function minval_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_5D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_5D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_5D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_5D = min( minval_double_st_5D, p_array(j) )
    end do

  end function minval_double_st_5D

  function minval_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_6D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_6D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_6D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_6D = min( minval_single_st_6D, p_array(j) )
    end do

  end function minval_single_st_6D

  function minval_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_6D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_6D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_6D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_6D = min( minval_double_st_6D, p_array(j) )
    end do

  end function minval_double_st_6D

  function minval_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: minval_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_single_st_7D =  huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_single_st_7D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_single_st_7D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_single_st_7D = min( minval_single_st_7D, p_array(j) )
    end do

  end function minval_single_st_7D

  function minval_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: minval_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`minval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`minval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        minval_double_st_7D =  huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`minval', 0, 3))
          minval_double_st_7D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`minval', 0, 3))
      minval_double_st_7D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      minval_double_st_7D = min( minval_double_st_7D, p_array(j) )
    end do

  end function minval_double_st_7D


  function maxval_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_1D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_1D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_1D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_1D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_1D = max( maxval_single_st_1D, p_array(j) )
    end do

  end function maxval_single_st_1D

  function maxval_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:), intent(in), target, contiguous :: array
    logical,         dimension (:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_1D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_1D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_1D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_1D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_1D = max( maxval_double_st_1D, p_array(j) )
    end do

  end function maxval_double_st_1D

  function maxval_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_2D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_2D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_2D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_2D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_2D = max( maxval_single_st_2D, p_array(j) )
    end do

  end function maxval_single_st_2D

  function maxval_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_2D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_2D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_2D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_2D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_2D = max( maxval_double_st_2D, p_array(j) )
    end do

  end function maxval_double_st_2D

  function maxval_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_3D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_3D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_3D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_3D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_3D = max( maxval_single_st_3D, p_array(j) )
    end do

  end function maxval_single_st_3D

  function maxval_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_3D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_3D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_3D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_3D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_3D = max( maxval_double_st_3D, p_array(j) )
    end do

  end function maxval_double_st_3D

  function maxval_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_4D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_4D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_4D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_4D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_4D = max( maxval_single_st_4D, p_array(j) )
    end do

  end function maxval_single_st_4D

  function maxval_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_4D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_4D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_4D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_4D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_4D = max( maxval_double_st_4D, p_array(j) )
    end do

  end function maxval_double_st_4D

  function maxval_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_5D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_5D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_5D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_5D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_5D = max( maxval_single_st_5D, p_array(j) )
    end do

  end function maxval_single_st_5D

  function maxval_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_5D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_5D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_5D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_5D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_5D = max( maxval_double_st_5D, p_array(j) )
    end do

  end function maxval_double_st_5D

  function maxval_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_6D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_6D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_6D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_6D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_6D = max( maxval_single_st_6D, p_array(j) )
    end do

  end function maxval_single_st_6D

  function maxval_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_6D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_6D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_6D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_6D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_6D = max( maxval_double_st_6D, p_array(j) )
    end do

  end function maxval_double_st_6D

  function maxval_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(single_st) :: maxval_single_st_7D
    ! Local declaration
    type(single_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(single_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_single_st_7D = - huge(0.)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_single_st_7D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_single_st_7D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_single_st_7D = max( maxval_single_st_7D, p_array(j) )
    end do

  end function maxval_single_st_7D

  function maxval_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,         dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(double_st) :: maxval_double_st_7D
    ! Local declaration
    type(double_st), dimension(:), pointer :: p_array
    logical,         dimension(:), pointer :: p_mask
    !`ifelse(regexp(`maxval', `val$'), `-1', `', `format(`type(double_st) val_%s', substr(`maxval', 0, 3))')'
    integer i, j

    p_array(1:size(array)) => array
    if (present(mask)) then
      if (count(mask) == 0) then
        maxval_double_st_7D = - huge(0.d0)
        return
      end if
      p_mask(1:size(mask)) => mask
      do i=1,size(p_array)
        if ( p_mask(i) ) then
          !format(`val_%s = p_array(i)', substr(`maxval', 0, 3))
          maxval_double_st_7D = p_array(i)
          exit
        end if
      end do
    else
      !format(`val_%s = p_array(1)', substr(`maxval', 0, 3))
      maxval_double_st_7D = p_array(1)
      i = 1
    end if 
    do j=i+1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(j) ) cycle 
      end if
      maxval_double_st_7D = max( maxval_double_st_7D, p_array(j) )
    end do

  end function maxval_double_st_7D




  function sum_single_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(single_st) :: sum_single_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          sum_single_st_dim_1D = sum(array, mask)
        else
          sum_single_st_dim_1D = sum(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        sum_single_st_dim_1D = 0. 
    end select
  end function sum_single_st_dim_1D

  function sum_double_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(double_st) :: sum_double_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          sum_double_st_dim_1D = sum(array, mask)
        else
          sum_double_st_dim_1D = sum(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        sum_double_st_dim_1D = 0.d0 
    end select
  end function sum_double_st_dim_1D

  function product_single_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(single_st) :: product_single_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          product_single_st_dim_1D = product(array, mask)
        else
          product_single_st_dim_1D = product(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        product_single_st_dim_1D = 1.
    end select
  end function product_single_st_dim_1D

  function product_double_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(double_st) :: product_double_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          product_double_st_dim_1D = product(array, mask)
        else
          product_double_st_dim_1D = product(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        product_double_st_dim_1D = 1.d0
    end select
  end function product_double_st_dim_1D

  function minloc_single_st_dim_1D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional,               intent(in) :: back
    integer :: minloc_single_st_dim_1D
    ! Local declaration
    integer, dimension(1) :: output 

    select case (dim)
      case (1)
        if (present(mask)) then
          output = minloc(array, mask, back)
        else
          output = minloc(array, back=back)
        end if
        minloc_single_st_dim_1D = output(1) 
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        minloc_single_st_dim_1D = 0
    end select
  end function minloc_single_st_dim_1D

  function minloc_double_st_dim_1D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional,               intent(in) :: back
    integer :: minloc_double_st_dim_1D
    ! Local declaration
    integer, dimension(1) :: output 

    select case (dim)
      case (1)
        if (present(mask)) then
          output = minloc(array, mask, back)
        else
          output = minloc(array, back=back)
        end if
        minloc_double_st_dim_1D = output(1) 
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        minloc_double_st_dim_1D = 0
    end select
  end function minloc_double_st_dim_1D

  function maxloc_single_st_dim_1D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional,               intent(in) :: back
    integer :: maxloc_single_st_dim_1D
    ! Local declaration
    integer, dimension(1) :: output 

    select case (dim)
      case (1)
        if (present(mask)) then
          output = maxloc(array, mask, back)
        else
          output = maxloc(array, back=back)
        end if
        maxloc_single_st_dim_1D = output(1) 
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        maxloc_single_st_dim_1D = 0
    end select
  end function maxloc_single_st_dim_1D

  function maxloc_double_st_dim_1D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    logical, optional,               intent(in) :: back
    integer :: maxloc_double_st_dim_1D
    ! Local declaration
    integer, dimension(1) :: output 

    select case (dim)
      case (1)
        if (present(mask)) then
          output = maxloc(array, mask, back)
        else
          output = maxloc(array, back=back)
        end if
        maxloc_double_st_dim_1D = output(1) 
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        maxloc_double_st_dim_1D = 0
    end select
  end function maxloc_double_st_dim_1D

  function minval_single_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(single_st) :: minval_single_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          minval_single_st_dim_1D = minval(array, mask)
        else
          minval_single_st_dim_1D = minval(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        minval_single_st_dim_1D = huge(array%x)
    end select
  end function minval_single_st_dim_1D

  function minval_double_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(double_st) :: minval_double_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          minval_double_st_dim_1D = minval(array, mask)
        else
          minval_double_st_dim_1D = minval(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        minval_double_st_dim_1D = huge(array%x)
    end select
  end function minval_double_st_dim_1D

  function maxval_single_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(single_st) :: maxval_single_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          maxval_single_st_dim_1D = maxval(array, mask)
        else
          maxval_single_st_dim_1D = maxval(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        maxval_single_st_dim_1D = -huge(array%x)
    end select
  end function maxval_single_st_dim_1D

  function maxval_double_st_dim_1D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:), intent(in) :: array
    integer,                         intent(in) :: dim
    logical, optional, dimension(:), intent(in) :: mask
    
    type(double_st) :: maxval_double_st_dim_1D
    

    select case (dim)
      case (1)
        if (present(mask)) then
          maxval_double_st_dim_1D = maxval(array, mask)
        else
          maxval_double_st_dim_1D = maxval(array)
        end if
        
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        maxval_double_st_dim_1D = -huge(array%x)
    end select
  end function maxval_double_st_dim_1D




  function sum_single_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(single_st), dimension(:), allocatable :: sum_single_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(sum_single_st_dim_2D,1)
          if (present(mask)) then
            sum_single_st_dim_2D(i1) = sum(array(:,i1), mask(:,i1))
          else
            sum_single_st_dim_2D(i1) = sum(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( sum_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(sum_single_st_dim_2D,1)
          if (present(mask)) then
            sum_single_st_dim_2D(i1) = sum(array(i1,:), mask(i1,:))
          else
            sum_single_st_dim_2D(i1) = sum(array(i1,:))
          end if
          
        end do
      case default
        allocate( sum_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_2D

  function sum_double_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(double_st), dimension(:), allocatable :: sum_double_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(sum_double_st_dim_2D,1)
          if (present(mask)) then
            sum_double_st_dim_2D(i1) = sum(array(:,i1), mask(:,i1))
          else
            sum_double_st_dim_2D(i1) = sum(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( sum_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(sum_double_st_dim_2D,1)
          if (present(mask)) then
            sum_double_st_dim_2D(i1) = sum(array(i1,:), mask(i1,:))
          else
            sum_double_st_dim_2D(i1) = sum(array(i1,:))
          end if
          
        end do
      case default
        allocate( sum_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_2D

  function product_single_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(single_st), dimension(:), allocatable :: product_single_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( product_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(product_single_st_dim_2D,1)
          if (present(mask)) then
            product_single_st_dim_2D(i1) = product(array(:,i1), mask(:,i1))
          else
            product_single_st_dim_2D(i1) = product(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( product_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(product_single_st_dim_2D,1)
          if (present(mask)) then
            product_single_st_dim_2D(i1) = product(array(i1,:), mask(i1,:))
          else
            product_single_st_dim_2D(i1) = product(array(i1,:))
          end if
          
        end do
      case default
        allocate( product_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_2D

  function product_double_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(double_st), dimension(:), allocatable :: product_double_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( product_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(product_double_st_dim_2D,1)
          if (present(mask)) then
            product_double_st_dim_2D(i1) = product(array(:,i1), mask(:,i1))
          else
            product_double_st_dim_2D(i1) = product(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( product_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(product_double_st_dim_2D,1)
          if (present(mask)) then
            product_double_st_dim_2D(i1) = product(array(i1,:), mask(i1,:))
          else
            product_double_st_dim_2D(i1) = product(array(i1,:))
          end if
          
        end do
      case default
        allocate( product_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_2D

  function minloc_single_st_dim_2D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    logical, optional,                 intent(in) :: back
    integer, dimension(:), allocatable :: minloc_single_st_dim_2D
    ! Local declaration
    integer, dimension(1) :: output
    integer i1

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(minloc_single_st_dim_2D,1)
          if (present(mask)) then
            output = minloc(array(:,i1), mask(:,i1), back)
          else
            output = minloc(array(:,i1), back=back)
          end if
          minloc_single_st_dim_2D(i1) = output(1) 
        end do
      case (2)
        allocate( minloc_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(minloc_single_st_dim_2D,1)
          if (present(mask)) then
            output = minloc(array(i1,:), mask(i1,:), back)
          else
            output = minloc(array(i1,:), back=back)
          end if
          minloc_single_st_dim_2D(i1) = output(1) 
        end do
      case default
        allocate( minloc_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_2D

  function minloc_double_st_dim_2D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    logical, optional,                 intent(in) :: back
    integer, dimension(:), allocatable :: minloc_double_st_dim_2D
    ! Local declaration
    integer, dimension(1) :: output
    integer i1

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(minloc_double_st_dim_2D,1)
          if (present(mask)) then
            output = minloc(array(:,i1), mask(:,i1), back)
          else
            output = minloc(array(:,i1), back=back)
          end if
          minloc_double_st_dim_2D(i1) = output(1) 
        end do
      case (2)
        allocate( minloc_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(minloc_double_st_dim_2D,1)
          if (present(mask)) then
            output = minloc(array(i1,:), mask(i1,:), back)
          else
            output = minloc(array(i1,:), back=back)
          end if
          minloc_double_st_dim_2D(i1) = output(1) 
        end do
      case default
        allocate( minloc_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_2D

  function maxloc_single_st_dim_2D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    logical, optional,                 intent(in) :: back
    integer, dimension(:), allocatable :: maxloc_single_st_dim_2D
    ! Local declaration
    integer, dimension(1) :: output
    integer i1

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(maxloc_single_st_dim_2D,1)
          if (present(mask)) then
            output = maxloc(array(:,i1), mask(:,i1), back)
          else
            output = maxloc(array(:,i1), back=back)
          end if
          maxloc_single_st_dim_2D(i1) = output(1) 
        end do
      case (2)
        allocate( maxloc_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(maxloc_single_st_dim_2D,1)
          if (present(mask)) then
            output = maxloc(array(i1,:), mask(i1,:), back)
          else
            output = maxloc(array(i1,:), back=back)
          end if
          maxloc_single_st_dim_2D(i1) = output(1) 
        end do
      case default
        allocate( maxloc_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_2D

  function maxloc_double_st_dim_2D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    logical, optional,                 intent(in) :: back
    integer, dimension(:), allocatable :: maxloc_double_st_dim_2D
    ! Local declaration
    integer, dimension(1) :: output
    integer i1

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(maxloc_double_st_dim_2D,1)
          if (present(mask)) then
            output = maxloc(array(:,i1), mask(:,i1), back)
          else
            output = maxloc(array(:,i1), back=back)
          end if
          maxloc_double_st_dim_2D(i1) = output(1) 
        end do
      case (2)
        allocate( maxloc_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(maxloc_double_st_dim_2D,1)
          if (present(mask)) then
            output = maxloc(array(i1,:), mask(i1,:), back)
          else
            output = maxloc(array(i1,:), back=back)
          end if
          maxloc_double_st_dim_2D(i1) = output(1) 
        end do
      case default
        allocate( maxloc_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_2D

  function minval_single_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(single_st), dimension(:), allocatable :: minval_single_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(minval_single_st_dim_2D,1)
          if (present(mask)) then
            minval_single_st_dim_2D(i1) = minval(array(:,i1), mask(:,i1))
          else
            minval_single_st_dim_2D(i1) = minval(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( minval_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(minval_single_st_dim_2D,1)
          if (present(mask)) then
            minval_single_st_dim_2D(i1) = minval(array(i1,:), mask(i1,:))
          else
            minval_single_st_dim_2D(i1) = minval(array(i1,:))
          end if
          
        end do
      case default
        allocate( minval_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_2D

  function minval_double_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(double_st), dimension(:), allocatable :: minval_double_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(minval_double_st_dim_2D,1)
          if (present(mask)) then
            minval_double_st_dim_2D(i1) = minval(array(:,i1), mask(:,i1))
          else
            minval_double_st_dim_2D(i1) = minval(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( minval_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(minval_double_st_dim_2D,1)
          if (present(mask)) then
            minval_double_st_dim_2D(i1) = minval(array(i1,:), mask(i1,:))
          else
            minval_double_st_dim_2D(i1) = minval(array(i1,:))
          end if
          
        end do
      case default
        allocate( minval_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_2D

  function maxval_single_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(single_st), dimension(:), allocatable :: maxval_single_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(maxval_single_st_dim_2D,1)
          if (present(mask)) then
            maxval_single_st_dim_2D(i1) = maxval(array(:,i1), mask(:,i1))
          else
            maxval_single_st_dim_2D(i1) = maxval(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( maxval_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(maxval_single_st_dim_2D,1)
          if (present(mask)) then
            maxval_single_st_dim_2D(i1) = maxval(array(i1,:), mask(i1,:))
          else
            maxval_single_st_dim_2D(i1) = maxval(array(i1,:))
          end if
          
        end do
      case default
        allocate( maxval_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_2D

  function maxval_double_st_dim_2D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:), intent(in) :: array
    integer,                           intent(in) :: dim
    logical, optional, dimension(:,:), intent(in) :: mask
    
    type(double_st), dimension(:), allocatable :: maxval_double_st_dim_2D
    ! Local declaration
    
    integer i1

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(maxval_double_st_dim_2D,1)
          if (present(mask)) then
            maxval_double_st_dim_2D(i1) = maxval(array(:,i1), mask(:,i1))
          else
            maxval_double_st_dim_2D(i1) = maxval(array(:,i1))
          end if
          
        end do
      case (2)
        allocate( maxval_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(maxval_double_st_dim_2D,1)
          if (present(mask)) then
            maxval_double_st_dim_2D(i1) = maxval(array(i1,:), mask(i1,:))
          else
            maxval_double_st_dim_2D(i1) = maxval(array(i1,:))
          end if
          
        end do
      case default
        allocate( maxval_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_2D




  function sum_single_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:), allocatable :: sum_single_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(sum_single_st_dim_3D,2)
          do i1=1,size(sum_single_st_dim_3D,1)
            if (present(mask)) then
              sum_single_st_dim_3D(i1,i2) = sum(array(:,i1,i2), mask(:,i1,i2))
            else
              sum_single_st_dim_3D(i1,i2) = sum(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( sum_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(sum_single_st_dim_3D,2)
          do i1=1,size(sum_single_st_dim_3D,1)
            if (present(mask)) then
              sum_single_st_dim_3D(i1,i2) = sum(array(i1,:,i2), mask(i1,:,i2))
            else
              sum_single_st_dim_3D(i1,i2) = sum(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( sum_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(sum_single_st_dim_3D,2)
          do i1=1,size(sum_single_st_dim_3D,1)
            if (present(mask)) then
              sum_single_st_dim_3D(i1,i2) = sum(array(i1,i2,:), mask(i1,i2,:))
            else
              sum_single_st_dim_3D(i1,i2) = sum(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( sum_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_3D

  function sum_double_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:), allocatable :: sum_double_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(sum_double_st_dim_3D,2)
          do i1=1,size(sum_double_st_dim_3D,1)
            if (present(mask)) then
              sum_double_st_dim_3D(i1,i2) = sum(array(:,i1,i2), mask(:,i1,i2))
            else
              sum_double_st_dim_3D(i1,i2) = sum(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( sum_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(sum_double_st_dim_3D,2)
          do i1=1,size(sum_double_st_dim_3D,1)
            if (present(mask)) then
              sum_double_st_dim_3D(i1,i2) = sum(array(i1,:,i2), mask(i1,:,i2))
            else
              sum_double_st_dim_3D(i1,i2) = sum(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( sum_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(sum_double_st_dim_3D,2)
          do i1=1,size(sum_double_st_dim_3D,1)
            if (present(mask)) then
              sum_double_st_dim_3D(i1,i2) = sum(array(i1,i2,:), mask(i1,i2,:))
            else
              sum_double_st_dim_3D(i1,i2) = sum(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( sum_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_3D

  function product_single_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:), allocatable :: product_single_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( product_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(product_single_st_dim_3D,2)
          do i1=1,size(product_single_st_dim_3D,1)
            if (present(mask)) then
              product_single_st_dim_3D(i1,i2) = product(array(:,i1,i2), mask(:,i1,i2))
            else
              product_single_st_dim_3D(i1,i2) = product(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( product_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(product_single_st_dim_3D,2)
          do i1=1,size(product_single_st_dim_3D,1)
            if (present(mask)) then
              product_single_st_dim_3D(i1,i2) = product(array(i1,:,i2), mask(i1,:,i2))
            else
              product_single_st_dim_3D(i1,i2) = product(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( product_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(product_single_st_dim_3D,2)
          do i1=1,size(product_single_st_dim_3D,1)
            if (present(mask)) then
              product_single_st_dim_3D(i1,i2) = product(array(i1,i2,:), mask(i1,i2,:))
            else
              product_single_st_dim_3D(i1,i2) = product(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( product_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_3D

  function product_double_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:), allocatable :: product_double_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( product_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(product_double_st_dim_3D,2)
          do i1=1,size(product_double_st_dim_3D,1)
            if (present(mask)) then
              product_double_st_dim_3D(i1,i2) = product(array(:,i1,i2), mask(:,i1,i2))
            else
              product_double_st_dim_3D(i1,i2) = product(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( product_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(product_double_st_dim_3D,2)
          do i1=1,size(product_double_st_dim_3D,1)
            if (present(mask)) then
              product_double_st_dim_3D(i1,i2) = product(array(i1,:,i2), mask(i1,:,i2))
            else
              product_double_st_dim_3D(i1,i2) = product(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( product_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(product_double_st_dim_3D,2)
          do i1=1,size(product_double_st_dim_3D,1)
            if (present(mask)) then
              product_double_st_dim_3D(i1,i2) = product(array(i1,i2,:), mask(i1,i2,:))
            else
              product_double_st_dim_3D(i1,i2) = product(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( product_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_3D

  function minloc_single_st_dim_3D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    logical, optional,                   intent(in) :: back
    integer, dimension(:,:), allocatable :: minloc_single_st_dim_3D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(minloc_single_st_dim_3D,2)
          do i1=1,size(minloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(:,i1,i2), mask(:,i1,i2), back)
            else
              output = minloc(array(:,i1,i2), back=back)
            end if
            minloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (2)
        allocate( minloc_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(minloc_single_st_dim_3D,2)
          do i1=1,size(minloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(i1,:,i2), mask(i1,:,i2), back)
            else
              output = minloc(array(i1,:,i2), back=back)
            end if
            minloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (3)
        allocate( minloc_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(minloc_single_st_dim_3D,2)
          do i1=1,size(minloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(i1,i2,:), mask(i1,i2,:), back)
            else
              output = minloc(array(i1,i2,:), back=back)
            end if
            minloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case default
        allocate( minloc_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_3D

  function minloc_double_st_dim_3D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    logical, optional,                   intent(in) :: back
    integer, dimension(:,:), allocatable :: minloc_double_st_dim_3D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(minloc_double_st_dim_3D,2)
          do i1=1,size(minloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(:,i1,i2), mask(:,i1,i2), back)
            else
              output = minloc(array(:,i1,i2), back=back)
            end if
            minloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (2)
        allocate( minloc_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(minloc_double_st_dim_3D,2)
          do i1=1,size(minloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(i1,:,i2), mask(i1,:,i2), back)
            else
              output = minloc(array(i1,:,i2), back=back)
            end if
            minloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (3)
        allocate( minloc_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(minloc_double_st_dim_3D,2)
          do i1=1,size(minloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = minloc(array(i1,i2,:), mask(i1,i2,:), back)
            else
              output = minloc(array(i1,i2,:), back=back)
            end if
            minloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case default
        allocate( minloc_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_3D

  function maxloc_single_st_dim_3D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    logical, optional,                   intent(in) :: back
    integer, dimension(:,:), allocatable :: maxloc_single_st_dim_3D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(maxloc_single_st_dim_3D,2)
          do i1=1,size(maxloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(:,i1,i2), mask(:,i1,i2), back)
            else
              output = maxloc(array(:,i1,i2), back=back)
            end if
            maxloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (2)
        allocate( maxloc_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(maxloc_single_st_dim_3D,2)
          do i1=1,size(maxloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(i1,:,i2), mask(i1,:,i2), back)
            else
              output = maxloc(array(i1,:,i2), back=back)
            end if
            maxloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (3)
        allocate( maxloc_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(maxloc_single_st_dim_3D,2)
          do i1=1,size(maxloc_single_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(i1,i2,:), mask(i1,i2,:), back)
            else
              output = maxloc(array(i1,i2,:), back=back)
            end if
            maxloc_single_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case default
        allocate( maxloc_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_3D

  function maxloc_double_st_dim_3D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    logical, optional,                   intent(in) :: back
    integer, dimension(:,:), allocatable :: maxloc_double_st_dim_3D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(maxloc_double_st_dim_3D,2)
          do i1=1,size(maxloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(:,i1,i2), mask(:,i1,i2), back)
            else
              output = maxloc(array(:,i1,i2), back=back)
            end if
            maxloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (2)
        allocate( maxloc_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(maxloc_double_st_dim_3D,2)
          do i1=1,size(maxloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(i1,:,i2), mask(i1,:,i2), back)
            else
              output = maxloc(array(i1,:,i2), back=back)
            end if
            maxloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case (3)
        allocate( maxloc_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(maxloc_double_st_dim_3D,2)
          do i1=1,size(maxloc_double_st_dim_3D,1)
            if (present(mask)) then
              output = maxloc(array(i1,i2,:), mask(i1,i2,:), back)
            else
              output = maxloc(array(i1,i2,:), back=back)
            end if
            maxloc_double_st_dim_3D(i1,i2) = output(1) 
          end do
        end do
      case default
        allocate( maxloc_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_3D

  function minval_single_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:), allocatable :: minval_single_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(minval_single_st_dim_3D,2)
          do i1=1,size(minval_single_st_dim_3D,1)
            if (present(mask)) then
              minval_single_st_dim_3D(i1,i2) = minval(array(:,i1,i2), mask(:,i1,i2))
            else
              minval_single_st_dim_3D(i1,i2) = minval(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( minval_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(minval_single_st_dim_3D,2)
          do i1=1,size(minval_single_st_dim_3D,1)
            if (present(mask)) then
              minval_single_st_dim_3D(i1,i2) = minval(array(i1,:,i2), mask(i1,:,i2))
            else
              minval_single_st_dim_3D(i1,i2) = minval(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( minval_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(minval_single_st_dim_3D,2)
          do i1=1,size(minval_single_st_dim_3D,1)
            if (present(mask)) then
              minval_single_st_dim_3D(i1,i2) = minval(array(i1,i2,:), mask(i1,i2,:))
            else
              minval_single_st_dim_3D(i1,i2) = minval(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( minval_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_3D

  function minval_double_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:), allocatable :: minval_double_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(minval_double_st_dim_3D,2)
          do i1=1,size(minval_double_st_dim_3D,1)
            if (present(mask)) then
              minval_double_st_dim_3D(i1,i2) = minval(array(:,i1,i2), mask(:,i1,i2))
            else
              minval_double_st_dim_3D(i1,i2) = minval(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( minval_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(minval_double_st_dim_3D,2)
          do i1=1,size(minval_double_st_dim_3D,1)
            if (present(mask)) then
              minval_double_st_dim_3D(i1,i2) = minval(array(i1,:,i2), mask(i1,:,i2))
            else
              minval_double_st_dim_3D(i1,i2) = minval(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( minval_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(minval_double_st_dim_3D,2)
          do i1=1,size(minval_double_st_dim_3D,1)
            if (present(mask)) then
              minval_double_st_dim_3D(i1,i2) = minval(array(i1,i2,:), mask(i1,i2,:))
            else
              minval_double_st_dim_3D(i1,i2) = minval(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( minval_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_3D

  function maxval_single_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:), allocatable :: maxval_single_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(maxval_single_st_dim_3D,2)
          do i1=1,size(maxval_single_st_dim_3D,1)
            if (present(mask)) then
              maxval_single_st_dim_3D(i1,i2) = maxval(array(:,i1,i2), mask(:,i1,i2))
            else
              maxval_single_st_dim_3D(i1,i2) = maxval(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( maxval_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(maxval_single_st_dim_3D,2)
          do i1=1,size(maxval_single_st_dim_3D,1)
            if (present(mask)) then
              maxval_single_st_dim_3D(i1,i2) = maxval(array(i1,:,i2), mask(i1,:,i2))
            else
              maxval_single_st_dim_3D(i1,i2) = maxval(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( maxval_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(maxval_single_st_dim_3D,2)
          do i1=1,size(maxval_single_st_dim_3D,1)
            if (present(mask)) then
              maxval_single_st_dim_3D(i1,i2) = maxval(array(i1,i2,:), mask(i1,i2,:))
            else
              maxval_single_st_dim_3D(i1,i2) = maxval(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( maxval_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_3D

  function maxval_double_st_dim_3D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:), intent(in) :: array
    integer,                             intent(in) :: dim
    logical, optional, dimension(:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:), allocatable :: maxval_double_st_dim_3D
    ! Local declaration
    
    integer i1, i2

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(maxval_double_st_dim_3D,2)
          do i1=1,size(maxval_double_st_dim_3D,1)
            if (present(mask)) then
              maxval_double_st_dim_3D(i1,i2) = maxval(array(:,i1,i2), mask(:,i1,i2))
            else
              maxval_double_st_dim_3D(i1,i2) = maxval(array(:,i1,i2))
            end if
            
          end do
        end do
      case (2)
        allocate( maxval_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(maxval_double_st_dim_3D,2)
          do i1=1,size(maxval_double_st_dim_3D,1)
            if (present(mask)) then
              maxval_double_st_dim_3D(i1,i2) = maxval(array(i1,:,i2), mask(i1,:,i2))
            else
              maxval_double_st_dim_3D(i1,i2) = maxval(array(i1,:,i2))
            end if
            
          end do
        end do
      case (3)
        allocate( maxval_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(maxval_double_st_dim_3D,2)
          do i1=1,size(maxval_double_st_dim_3D,1)
            if (present(mask)) then
              maxval_double_st_dim_3D(i1,i2) = maxval(array(i1,i2,:), mask(i1,i2,:))
            else
              maxval_double_st_dim_3D(i1,i2) = maxval(array(i1,i2,:))
            end if
            
          end do
        end do
      case default
        allocate( maxval_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_3D




  function sum_single_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:), allocatable :: sum_single_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_single_st_dim_4D,3)
          do i2=1,size(sum_single_st_dim_4D,2)
            do i1=1,size(sum_single_st_dim_4D,1)
              if (present(mask)) then
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( sum_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_single_st_dim_4D,3)
          do i2=1,size(sum_single_st_dim_4D,2)
            do i1=1,size(sum_single_st_dim_4D,1)
              if (present(mask)) then
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( sum_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(sum_single_st_dim_4D,3)
          do i2=1,size(sum_single_st_dim_4D,2)
            do i1=1,size(sum_single_st_dim_4D,1)
              if (present(mask)) then
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( sum_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(sum_single_st_dim_4D,3)
          do i2=1,size(sum_single_st_dim_4D,2)
            do i1=1,size(sum_single_st_dim_4D,1)
              if (present(mask)) then
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                sum_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( sum_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_4D

  function sum_double_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:), allocatable :: sum_double_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_double_st_dim_4D,3)
          do i2=1,size(sum_double_st_dim_4D,2)
            do i1=1,size(sum_double_st_dim_4D,1)
              if (present(mask)) then
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( sum_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_double_st_dim_4D,3)
          do i2=1,size(sum_double_st_dim_4D,2)
            do i1=1,size(sum_double_st_dim_4D,1)
              if (present(mask)) then
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( sum_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(sum_double_st_dim_4D,3)
          do i2=1,size(sum_double_st_dim_4D,2)
            do i1=1,size(sum_double_st_dim_4D,1)
              if (present(mask)) then
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( sum_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(sum_double_st_dim_4D,3)
          do i2=1,size(sum_double_st_dim_4D,2)
            do i1=1,size(sum_double_st_dim_4D,1)
              if (present(mask)) then
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                sum_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( sum_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_4D

  function product_single_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:), allocatable :: product_single_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( product_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(product_single_st_dim_4D,3)
          do i2=1,size(product_single_st_dim_4D,2)
            do i1=1,size(product_single_st_dim_4D,1)
              if (present(mask)) then
                product_single_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                product_single_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( product_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(product_single_st_dim_4D,3)
          do i2=1,size(product_single_st_dim_4D,2)
            do i1=1,size(product_single_st_dim_4D,1)
              if (present(mask)) then
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( product_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(product_single_st_dim_4D,3)
          do i2=1,size(product_single_st_dim_4D,2)
            do i1=1,size(product_single_st_dim_4D,1)
              if (present(mask)) then
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( product_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(product_single_st_dim_4D,3)
          do i2=1,size(product_single_st_dim_4D,2)
            do i1=1,size(product_single_st_dim_4D,1)
              if (present(mask)) then
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                product_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( product_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_4D

  function product_double_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:), allocatable :: product_double_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( product_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(product_double_st_dim_4D,3)
          do i2=1,size(product_double_st_dim_4D,2)
            do i1=1,size(product_double_st_dim_4D,1)
              if (present(mask)) then
                product_double_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                product_double_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( product_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(product_double_st_dim_4D,3)
          do i2=1,size(product_double_st_dim_4D,2)
            do i1=1,size(product_double_st_dim_4D,1)
              if (present(mask)) then
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( product_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(product_double_st_dim_4D,3)
          do i2=1,size(product_double_st_dim_4D,2)
            do i1=1,size(product_double_st_dim_4D,1)
              if (present(mask)) then
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( product_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(product_double_st_dim_4D,3)
          do i2=1,size(product_double_st_dim_4D,2)
            do i1=1,size(product_double_st_dim_4D,1)
              if (present(mask)) then
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                product_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( product_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_4D

  function minloc_single_st_dim_4D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    logical, optional,                     intent(in) :: back
    integer, dimension(:,:,:), allocatable :: minloc_single_st_dim_4D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(minloc_single_st_dim_4D,3)
          do i2=1,size(minloc_single_st_dim_4D,2)
            do i1=1,size(minloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(:,i1,i2,i3), mask(:,i1,i2,i3), back)
              else
                output = minloc(array(:,i1,i2,i3), back=back)
              end if
              minloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (2)
        allocate( minloc_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(minloc_single_st_dim_4D,3)
          do i2=1,size(minloc_single_st_dim_4D,2)
            do i1=1,size(minloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,:,i2,i3), mask(i1,:,i2,i3), back)
              else
                output = minloc(array(i1,:,i2,i3), back=back)
              end if
              minloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (3)
        allocate( minloc_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(minloc_single_st_dim_4D,3)
          do i2=1,size(minloc_single_st_dim_4D,2)
            do i1=1,size(minloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,i2,:,i3), mask(i1,i2,:,i3), back)
              else
                output = minloc(array(i1,i2,:,i3), back=back)
              end if
              minloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (4)
        allocate( minloc_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(minloc_single_st_dim_4D,3)
          do i2=1,size(minloc_single_st_dim_4D,2)
            do i1=1,size(minloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,i2,i3,:), mask(i1,i2,i3,:), back)
              else
                output = minloc(array(i1,i2,i3,:), back=back)
              end if
              minloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case default
        allocate( minloc_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_4D

  function minloc_double_st_dim_4D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    logical, optional,                     intent(in) :: back
    integer, dimension(:,:,:), allocatable :: minloc_double_st_dim_4D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(minloc_double_st_dim_4D,3)
          do i2=1,size(minloc_double_st_dim_4D,2)
            do i1=1,size(minloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(:,i1,i2,i3), mask(:,i1,i2,i3), back)
              else
                output = minloc(array(:,i1,i2,i3), back=back)
              end if
              minloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (2)
        allocate( minloc_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(minloc_double_st_dim_4D,3)
          do i2=1,size(minloc_double_st_dim_4D,2)
            do i1=1,size(minloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,:,i2,i3), mask(i1,:,i2,i3), back)
              else
                output = minloc(array(i1,:,i2,i3), back=back)
              end if
              minloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (3)
        allocate( minloc_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(minloc_double_st_dim_4D,3)
          do i2=1,size(minloc_double_st_dim_4D,2)
            do i1=1,size(minloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,i2,:,i3), mask(i1,i2,:,i3), back)
              else
                output = minloc(array(i1,i2,:,i3), back=back)
              end if
              minloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (4)
        allocate( minloc_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(minloc_double_st_dim_4D,3)
          do i2=1,size(minloc_double_st_dim_4D,2)
            do i1=1,size(minloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = minloc(array(i1,i2,i3,:), mask(i1,i2,i3,:), back)
              else
                output = minloc(array(i1,i2,i3,:), back=back)
              end if
              minloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case default
        allocate( minloc_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_4D

  function maxloc_single_st_dim_4D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    logical, optional,                     intent(in) :: back
    integer, dimension(:,:,:), allocatable :: maxloc_single_st_dim_4D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(maxloc_single_st_dim_4D,3)
          do i2=1,size(maxloc_single_st_dim_4D,2)
            do i1=1,size(maxloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(:,i1,i2,i3), mask(:,i1,i2,i3), back)
              else
                output = maxloc(array(:,i1,i2,i3), back=back)
              end if
              maxloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (2)
        allocate( maxloc_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(maxloc_single_st_dim_4D,3)
          do i2=1,size(maxloc_single_st_dim_4D,2)
            do i1=1,size(maxloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,:,i2,i3), mask(i1,:,i2,i3), back)
              else
                output = maxloc(array(i1,:,i2,i3), back=back)
              end if
              maxloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (3)
        allocate( maxloc_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(maxloc_single_st_dim_4D,3)
          do i2=1,size(maxloc_single_st_dim_4D,2)
            do i1=1,size(maxloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,i2,:,i3), mask(i1,i2,:,i3), back)
              else
                output = maxloc(array(i1,i2,:,i3), back=back)
              end if
              maxloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (4)
        allocate( maxloc_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(maxloc_single_st_dim_4D,3)
          do i2=1,size(maxloc_single_st_dim_4D,2)
            do i1=1,size(maxloc_single_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,i2,i3,:), mask(i1,i2,i3,:), back)
              else
                output = maxloc(array(i1,i2,i3,:), back=back)
              end if
              maxloc_single_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case default
        allocate( maxloc_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_4D

  function maxloc_double_st_dim_4D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    logical, optional,                     intent(in) :: back
    integer, dimension(:,:,:), allocatable :: maxloc_double_st_dim_4D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(maxloc_double_st_dim_4D,3)
          do i2=1,size(maxloc_double_st_dim_4D,2)
            do i1=1,size(maxloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(:,i1,i2,i3), mask(:,i1,i2,i3), back)
              else
                output = maxloc(array(:,i1,i2,i3), back=back)
              end if
              maxloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (2)
        allocate( maxloc_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(maxloc_double_st_dim_4D,3)
          do i2=1,size(maxloc_double_st_dim_4D,2)
            do i1=1,size(maxloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,:,i2,i3), mask(i1,:,i2,i3), back)
              else
                output = maxloc(array(i1,:,i2,i3), back=back)
              end if
              maxloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (3)
        allocate( maxloc_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(maxloc_double_st_dim_4D,3)
          do i2=1,size(maxloc_double_st_dim_4D,2)
            do i1=1,size(maxloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,i2,:,i3), mask(i1,i2,:,i3), back)
              else
                output = maxloc(array(i1,i2,:,i3), back=back)
              end if
              maxloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case (4)
        allocate( maxloc_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(maxloc_double_st_dim_4D,3)
          do i2=1,size(maxloc_double_st_dim_4D,2)
            do i1=1,size(maxloc_double_st_dim_4D,1)
              if (present(mask)) then
                output = maxloc(array(i1,i2,i3,:), mask(i1,i2,i3,:), back)
              else
                output = maxloc(array(i1,i2,i3,:), back=back)
              end if
              maxloc_double_st_dim_4D(i1,i2,i3) = output(1) 
            end do
          end do
        end do
      case default
        allocate( maxloc_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_4D

  function minval_single_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:), allocatable :: minval_single_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(minval_single_st_dim_4D,3)
          do i2=1,size(minval_single_st_dim_4D,2)
            do i1=1,size(minval_single_st_dim_4D,1)
              if (present(mask)) then
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( minval_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(minval_single_st_dim_4D,3)
          do i2=1,size(minval_single_st_dim_4D,2)
            do i1=1,size(minval_single_st_dim_4D,1)
              if (present(mask)) then
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( minval_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(minval_single_st_dim_4D,3)
          do i2=1,size(minval_single_st_dim_4D,2)
            do i1=1,size(minval_single_st_dim_4D,1)
              if (present(mask)) then
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( minval_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(minval_single_st_dim_4D,3)
          do i2=1,size(minval_single_st_dim_4D,2)
            do i1=1,size(minval_single_st_dim_4D,1)
              if (present(mask)) then
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                minval_single_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( minval_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_4D

  function minval_double_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:), allocatable :: minval_double_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(minval_double_st_dim_4D,3)
          do i2=1,size(minval_double_st_dim_4D,2)
            do i1=1,size(minval_double_st_dim_4D,1)
              if (present(mask)) then
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( minval_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(minval_double_st_dim_4D,3)
          do i2=1,size(minval_double_st_dim_4D,2)
            do i1=1,size(minval_double_st_dim_4D,1)
              if (present(mask)) then
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( minval_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(minval_double_st_dim_4D,3)
          do i2=1,size(minval_double_st_dim_4D,2)
            do i1=1,size(minval_double_st_dim_4D,1)
              if (present(mask)) then
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( minval_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(minval_double_st_dim_4D,3)
          do i2=1,size(minval_double_st_dim_4D,2)
            do i1=1,size(minval_double_st_dim_4D,1)
              if (present(mask)) then
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                minval_double_st_dim_4D(i1,i2,i3) = minval(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( minval_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_4D

  function maxval_single_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:), allocatable :: maxval_single_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(maxval_single_st_dim_4D,3)
          do i2=1,size(maxval_single_st_dim_4D,2)
            do i1=1,size(maxval_single_st_dim_4D,1)
              if (present(mask)) then
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( maxval_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(maxval_single_st_dim_4D,3)
          do i2=1,size(maxval_single_st_dim_4D,2)
            do i1=1,size(maxval_single_st_dim_4D,1)
              if (present(mask)) then
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( maxval_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(maxval_single_st_dim_4D,3)
          do i2=1,size(maxval_single_st_dim_4D,2)
            do i1=1,size(maxval_single_st_dim_4D,1)
              if (present(mask)) then
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( maxval_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(maxval_single_st_dim_4D,3)
          do i2=1,size(maxval_single_st_dim_4D,2)
            do i1=1,size(maxval_single_st_dim_4D,1)
              if (present(mask)) then
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                maxval_single_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( maxval_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_4D

  function maxval_double_st_dim_4D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:), intent(in) :: array
    integer,                               intent(in) :: dim
    logical, optional, dimension(:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:), allocatable :: maxval_double_st_dim_4D
    ! Local declaration
    
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(maxval_double_st_dim_4D,3)
          do i2=1,size(maxval_double_st_dim_4D,2)
            do i1=1,size(maxval_double_st_dim_4D,1)
              if (present(mask)) then
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(:,i1,i2,i3))
              end if
              
            end do
          end do
        end do
      case (2)
        allocate( maxval_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(maxval_double_st_dim_4D,3)
          do i2=1,size(maxval_double_st_dim_4D,2)
            do i1=1,size(maxval_double_st_dim_4D,1)
              if (present(mask)) then
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,:,i2,i3))
              end if
              
            end do
          end do
        end do
      case (3)
        allocate( maxval_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(maxval_double_st_dim_4D,3)
          do i2=1,size(maxval_double_st_dim_4D,2)
            do i1=1,size(maxval_double_st_dim_4D,1)
              if (present(mask)) then
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,:,i3))
              end if
              
            end do
          end do
        end do
      case (4)
        allocate( maxval_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(maxval_double_st_dim_4D,3)
          do i2=1,size(maxval_double_st_dim_4D,2)
            do i1=1,size(maxval_double_st_dim_4D,1)
              if (present(mask)) then
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                maxval_double_st_dim_4D(i1,i2,i3) = maxval(array(i1,i2,i3,:))
              end if
              
            end do
          end do
        end do
      case default
        allocate( maxval_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_4D




  function sum_single_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:), allocatable :: sum_single_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_single_st_dim_5D,4)
          do i3=1,size(sum_single_st_dim_5D,3)
            do i2=1,size(sum_single_st_dim_5D,2)
              do i1=1,size(sum_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_single_st_dim_5D,4)
          do i3=1,size(sum_single_st_dim_5D,3)
            do i2=1,size(sum_single_st_dim_5D,2)
              do i1=1,size(sum_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_single_st_dim_5D,4)
          do i3=1,size(sum_single_st_dim_5D,3)
            do i2=1,size(sum_single_st_dim_5D,2)
              do i1=1,size(sum_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(sum_single_st_dim_5D,4)
          do i3=1,size(sum_single_st_dim_5D,3)
            do i2=1,size(sum_single_st_dim_5D,2)
              do i1=1,size(sum_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(sum_single_st_dim_5D,4)
          do i3=1,size(sum_single_st_dim_5D,3)
            do i2=1,size(sum_single_st_dim_5D,2)
              do i1=1,size(sum_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  sum_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( sum_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_5D

  function sum_double_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:), allocatable :: sum_double_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_double_st_dim_5D,4)
          do i3=1,size(sum_double_st_dim_5D,3)
            do i2=1,size(sum_double_st_dim_5D,2)
              do i1=1,size(sum_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_double_st_dim_5D,4)
          do i3=1,size(sum_double_st_dim_5D,3)
            do i2=1,size(sum_double_st_dim_5D,2)
              do i1=1,size(sum_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_double_st_dim_5D,4)
          do i3=1,size(sum_double_st_dim_5D,3)
            do i2=1,size(sum_double_st_dim_5D,2)
              do i1=1,size(sum_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(sum_double_st_dim_5D,4)
          do i3=1,size(sum_double_st_dim_5D,3)
            do i2=1,size(sum_double_st_dim_5D,2)
              do i1=1,size(sum_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(sum_double_st_dim_5D,4)
          do i3=1,size(sum_double_st_dim_5D,3)
            do i2=1,size(sum_double_st_dim_5D,2)
              do i1=1,size(sum_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  sum_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( sum_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_5D

  function product_single_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:), allocatable :: product_single_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( product_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_single_st_dim_5D,4)
          do i3=1,size(product_single_st_dim_5D,3)
            do i2=1,size(product_single_st_dim_5D,2)
              do i1=1,size(product_single_st_dim_5D,1)
                if (present(mask)) then
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_single_st_dim_5D,4)
          do i3=1,size(product_single_st_dim_5D,3)
            do i2=1,size(product_single_st_dim_5D,2)
              do i1=1,size(product_single_st_dim_5D,1)
                if (present(mask)) then
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(product_single_st_dim_5D,4)
          do i3=1,size(product_single_st_dim_5D,3)
            do i2=1,size(product_single_st_dim_5D,2)
              do i1=1,size(product_single_st_dim_5D,1)
                if (present(mask)) then
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(product_single_st_dim_5D,4)
          do i3=1,size(product_single_st_dim_5D,3)
            do i2=1,size(product_single_st_dim_5D,2)
              do i1=1,size(product_single_st_dim_5D,1)
                if (present(mask)) then
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(product_single_st_dim_5D,4)
          do i3=1,size(product_single_st_dim_5D,3)
            do i2=1,size(product_single_st_dim_5D,2)
              do i1=1,size(product_single_st_dim_5D,1)
                if (present(mask)) then
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  product_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( product_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_5D

  function product_double_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:), allocatable :: product_double_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( product_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_double_st_dim_5D,4)
          do i3=1,size(product_double_st_dim_5D,3)
            do i2=1,size(product_double_st_dim_5D,2)
              do i1=1,size(product_double_st_dim_5D,1)
                if (present(mask)) then
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_double_st_dim_5D,4)
          do i3=1,size(product_double_st_dim_5D,3)
            do i2=1,size(product_double_st_dim_5D,2)
              do i1=1,size(product_double_st_dim_5D,1)
                if (present(mask)) then
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(product_double_st_dim_5D,4)
          do i3=1,size(product_double_st_dim_5D,3)
            do i2=1,size(product_double_st_dim_5D,2)
              do i1=1,size(product_double_st_dim_5D,1)
                if (present(mask)) then
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(product_double_st_dim_5D,4)
          do i3=1,size(product_double_st_dim_5D,3)
            do i2=1,size(product_double_st_dim_5D,2)
              do i1=1,size(product_double_st_dim_5D,1)
                if (present(mask)) then
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(product_double_st_dim_5D,4)
          do i3=1,size(product_double_st_dim_5D,3)
            do i2=1,size(product_double_st_dim_5D,2)
              do i1=1,size(product_double_st_dim_5D,1)
                if (present(mask)) then
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  product_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( product_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_5D

  function minloc_single_st_dim_5D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    logical, optional,                       intent(in) :: back
    integer, dimension(:,:,:,:), allocatable :: minloc_single_st_dim_5D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_single_st_dim_5D,4)
          do i3=1,size(minloc_single_st_dim_5D,3)
            do i2=1,size(minloc_single_st_dim_5D,2)
              do i1=1,size(minloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4), back)
                else
                  output = minloc(array(:,i1,i2,i3,i4), back=back)
                end if
                minloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_single_st_dim_5D,4)
          do i3=1,size(minloc_single_st_dim_5D,3)
            do i2=1,size(minloc_single_st_dim_5D,2)
              do i1=1,size(minloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4), back)
                else
                  output = minloc(array(i1,:,i2,i3,i4), back=back)
                end if
                minloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_single_st_dim_5D,4)
          do i3=1,size(minloc_single_st_dim_5D,3)
            do i2=1,size(minloc_single_st_dim_5D,2)
              do i1=1,size(minloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4), back)
                else
                  output = minloc(array(i1,i2,:,i3,i4), back=back)
                end if
                minloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(minloc_single_st_dim_5D,4)
          do i3=1,size(minloc_single_st_dim_5D,3)
            do i2=1,size(minloc_single_st_dim_5D,2)
              do i1=1,size(minloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4), back)
                else
                  output = minloc(array(i1,i2,i3,:,i4), back=back)
                end if
                minloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(minloc_single_st_dim_5D,4)
          do i3=1,size(minloc_single_st_dim_5D,3)
            do i2=1,size(minloc_single_st_dim_5D,2)
              do i1=1,size(minloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:), back)
                else
                  output = minloc(array(i1,i2,i3,i4,:), back=back)
                end if
                minloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_5D

  function minloc_double_st_dim_5D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    logical, optional,                       intent(in) :: back
    integer, dimension(:,:,:,:), allocatable :: minloc_double_st_dim_5D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_double_st_dim_5D,4)
          do i3=1,size(minloc_double_st_dim_5D,3)
            do i2=1,size(minloc_double_st_dim_5D,2)
              do i1=1,size(minloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4), back)
                else
                  output = minloc(array(:,i1,i2,i3,i4), back=back)
                end if
                minloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_double_st_dim_5D,4)
          do i3=1,size(minloc_double_st_dim_5D,3)
            do i2=1,size(minloc_double_st_dim_5D,2)
              do i1=1,size(minloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4), back)
                else
                  output = minloc(array(i1,:,i2,i3,i4), back=back)
                end if
                minloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(minloc_double_st_dim_5D,4)
          do i3=1,size(minloc_double_st_dim_5D,3)
            do i2=1,size(minloc_double_st_dim_5D,2)
              do i1=1,size(minloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4), back)
                else
                  output = minloc(array(i1,i2,:,i3,i4), back=back)
                end if
                minloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(minloc_double_st_dim_5D,4)
          do i3=1,size(minloc_double_st_dim_5D,3)
            do i2=1,size(minloc_double_st_dim_5D,2)
              do i1=1,size(minloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4), back)
                else
                  output = minloc(array(i1,i2,i3,:,i4), back=back)
                end if
                minloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(minloc_double_st_dim_5D,4)
          do i3=1,size(minloc_double_st_dim_5D,3)
            do i2=1,size(minloc_double_st_dim_5D,2)
              do i1=1,size(minloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = minloc(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:), back)
                else
                  output = minloc(array(i1,i2,i3,i4,:), back=back)
                end if
                minloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_5D

  function maxloc_single_st_dim_5D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    logical, optional,                       intent(in) :: back
    integer, dimension(:,:,:,:), allocatable :: maxloc_single_st_dim_5D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_single_st_dim_5D,4)
          do i3=1,size(maxloc_single_st_dim_5D,3)
            do i2=1,size(maxloc_single_st_dim_5D,2)
              do i1=1,size(maxloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4), back)
                else
                  output = maxloc(array(:,i1,i2,i3,i4), back=back)
                end if
                maxloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_single_st_dim_5D,4)
          do i3=1,size(maxloc_single_st_dim_5D,3)
            do i2=1,size(maxloc_single_st_dim_5D,2)
              do i1=1,size(maxloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4), back)
                else
                  output = maxloc(array(i1,:,i2,i3,i4), back=back)
                end if
                maxloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_single_st_dim_5D,4)
          do i3=1,size(maxloc_single_st_dim_5D,3)
            do i2=1,size(maxloc_single_st_dim_5D,2)
              do i1=1,size(maxloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4), back)
                else
                  output = maxloc(array(i1,i2,:,i3,i4), back=back)
                end if
                maxloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(maxloc_single_st_dim_5D,4)
          do i3=1,size(maxloc_single_st_dim_5D,3)
            do i2=1,size(maxloc_single_st_dim_5D,2)
              do i1=1,size(maxloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4), back)
                else
                  output = maxloc(array(i1,i2,i3,:,i4), back=back)
                end if
                maxloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(maxloc_single_st_dim_5D,4)
          do i3=1,size(maxloc_single_st_dim_5D,3)
            do i2=1,size(maxloc_single_st_dim_5D,2)
              do i1=1,size(maxloc_single_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:), back)
                else
                  output = maxloc(array(i1,i2,i3,i4,:), back=back)
                end if
                maxloc_single_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_5D

  function maxloc_double_st_dim_5D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    logical, optional,                       intent(in) :: back
    integer, dimension(:,:,:,:), allocatable :: maxloc_double_st_dim_5D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_double_st_dim_5D,4)
          do i3=1,size(maxloc_double_st_dim_5D,3)
            do i2=1,size(maxloc_double_st_dim_5D,2)
              do i1=1,size(maxloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4), back)
                else
                  output = maxloc(array(:,i1,i2,i3,i4), back=back)
                end if
                maxloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_double_st_dim_5D,4)
          do i3=1,size(maxloc_double_st_dim_5D,3)
            do i2=1,size(maxloc_double_st_dim_5D,2)
              do i1=1,size(maxloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4), back)
                else
                  output = maxloc(array(i1,:,i2,i3,i4), back=back)
                end if
                maxloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(maxloc_double_st_dim_5D,4)
          do i3=1,size(maxloc_double_st_dim_5D,3)
            do i2=1,size(maxloc_double_st_dim_5D,2)
              do i1=1,size(maxloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4), back)
                else
                  output = maxloc(array(i1,i2,:,i3,i4), back=back)
                end if
                maxloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(maxloc_double_st_dim_5D,4)
          do i3=1,size(maxloc_double_st_dim_5D,3)
            do i2=1,size(maxloc_double_st_dim_5D,2)
              do i1=1,size(maxloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4), back)
                else
                  output = maxloc(array(i1,i2,i3,:,i4), back=back)
                end if
                maxloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(maxloc_double_st_dim_5D,4)
          do i3=1,size(maxloc_double_st_dim_5D,3)
            do i2=1,size(maxloc_double_st_dim_5D,2)
              do i1=1,size(maxloc_double_st_dim_5D,1)
                if (present(mask)) then
                  output = maxloc(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:), back)
                else
                  output = maxloc(array(i1,i2,i3,i4,:), back=back)
                end if
                maxloc_double_st_dim_5D(i1,i2,i3,i4) = output(1) 
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_5D

  function minval_single_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:), allocatable :: minval_single_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_single_st_dim_5D,4)
          do i3=1,size(minval_single_st_dim_5D,3)
            do i2=1,size(minval_single_st_dim_5D,2)
              do i1=1,size(minval_single_st_dim_5D,1)
                if (present(mask)) then
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_single_st_dim_5D,4)
          do i3=1,size(minval_single_st_dim_5D,3)
            do i2=1,size(minval_single_st_dim_5D,2)
              do i1=1,size(minval_single_st_dim_5D,1)
                if (present(mask)) then
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_single_st_dim_5D,4)
          do i3=1,size(minval_single_st_dim_5D,3)
            do i2=1,size(minval_single_st_dim_5D,2)
              do i1=1,size(minval_single_st_dim_5D,1)
                if (present(mask)) then
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(minval_single_st_dim_5D,4)
          do i3=1,size(minval_single_st_dim_5D,3)
            do i2=1,size(minval_single_st_dim_5D,2)
              do i1=1,size(minval_single_st_dim_5D,1)
                if (present(mask)) then
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(minval_single_st_dim_5D,4)
          do i3=1,size(minval_single_st_dim_5D,3)
            do i2=1,size(minval_single_st_dim_5D,2)
              do i1=1,size(minval_single_st_dim_5D,1)
                if (present(mask)) then
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  minval_single_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( minval_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_5D

  function minval_double_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:), allocatable :: minval_double_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_double_st_dim_5D,4)
          do i3=1,size(minval_double_st_dim_5D,3)
            do i2=1,size(minval_double_st_dim_5D,2)
              do i1=1,size(minval_double_st_dim_5D,1)
                if (present(mask)) then
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_double_st_dim_5D,4)
          do i3=1,size(minval_double_st_dim_5D,3)
            do i2=1,size(minval_double_st_dim_5D,2)
              do i1=1,size(minval_double_st_dim_5D,1)
                if (present(mask)) then
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(minval_double_st_dim_5D,4)
          do i3=1,size(minval_double_st_dim_5D,3)
            do i2=1,size(minval_double_st_dim_5D,2)
              do i1=1,size(minval_double_st_dim_5D,1)
                if (present(mask)) then
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(minval_double_st_dim_5D,4)
          do i3=1,size(minval_double_st_dim_5D,3)
            do i2=1,size(minval_double_st_dim_5D,2)
              do i1=1,size(minval_double_st_dim_5D,1)
                if (present(mask)) then
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(minval_double_st_dim_5D,4)
          do i3=1,size(minval_double_st_dim_5D,3)
            do i2=1,size(minval_double_st_dim_5D,2)
              do i1=1,size(minval_double_st_dim_5D,1)
                if (present(mask)) then
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  minval_double_st_dim_5D(i1,i2,i3,i4) = minval(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( minval_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_5D

  function maxval_single_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:), allocatable :: maxval_single_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_single_st_dim_5D,4)
          do i3=1,size(maxval_single_st_dim_5D,3)
            do i2=1,size(maxval_single_st_dim_5D,2)
              do i1=1,size(maxval_single_st_dim_5D,1)
                if (present(mask)) then
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_single_st_dim_5D,4)
          do i3=1,size(maxval_single_st_dim_5D,3)
            do i2=1,size(maxval_single_st_dim_5D,2)
              do i1=1,size(maxval_single_st_dim_5D,1)
                if (present(mask)) then
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_single_st_dim_5D,4)
          do i3=1,size(maxval_single_st_dim_5D,3)
            do i2=1,size(maxval_single_st_dim_5D,2)
              do i1=1,size(maxval_single_st_dim_5D,1)
                if (present(mask)) then
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(maxval_single_st_dim_5D,4)
          do i3=1,size(maxval_single_st_dim_5D,3)
            do i2=1,size(maxval_single_st_dim_5D,2)
              do i1=1,size(maxval_single_st_dim_5D,1)
                if (present(mask)) then
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(maxval_single_st_dim_5D,4)
          do i3=1,size(maxval_single_st_dim_5D,3)
            do i2=1,size(maxval_single_st_dim_5D,2)
              do i1=1,size(maxval_single_st_dim_5D,1)
                if (present(mask)) then
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  maxval_single_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_5D

  function maxval_double_st_dim_5D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:), intent(in) :: array
    integer,                                 intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:), allocatable :: maxval_double_st_dim_5D
    ! Local declaration
    
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_double_st_dim_5D,4)
          do i3=1,size(maxval_double_st_dim_5D,3)
            do i2=1,size(maxval_double_st_dim_5D,2)
              do i1=1,size(maxval_double_st_dim_5D,1)
                if (present(mask)) then
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(:,i1,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_double_st_dim_5D,4)
          do i3=1,size(maxval_double_st_dim_5D,3)
            do i2=1,size(maxval_double_st_dim_5D,2)
              do i1=1,size(maxval_double_st_dim_5D,1)
                if (present(mask)) then
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,:,i2,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(maxval_double_st_dim_5D,4)
          do i3=1,size(maxval_double_st_dim_5D,3)
            do i2=1,size(maxval_double_st_dim_5D,2)
              do i1=1,size(maxval_double_st_dim_5D,1)
                if (present(mask)) then
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,:,i3,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5)) ) 
        do i4=1,size(maxval_double_st_dim_5D,4)
          do i3=1,size(maxval_double_st_dim_5D,3)
            do i2=1,size(maxval_double_st_dim_5D,2)
              do i1=1,size(maxval_double_st_dim_5D,1)
                if (present(mask)) then
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,:,i4))
                end if
                
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(maxval_double_st_dim_5D,4)
          do i3=1,size(maxval_double_st_dim_5D,3)
            do i2=1,size(maxval_double_st_dim_5D,2)
              do i1=1,size(maxval_double_st_dim_5D,1)
                if (present(mask)) then
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  maxval_double_st_dim_5D(i1,i2,i3,i4) = maxval(array(i1,i2,i3,i4,:))
                end if
                
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_5D




  function sum_single_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:), allocatable :: sum_single_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,6)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5)) )
        do i5=1,size(sum_single_st_dim_6D,5)
          do i4=1,size(sum_single_st_dim_6D,4)
            do i3=1,size(sum_single_st_dim_6D,3)
              do i2=1,size(sum_single_st_dim_6D,2)
                do i1=1,size(sum_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    sum_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_6D

  function sum_double_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:), allocatable :: sum_double_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,5), size(array,6)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,6)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5)) )
        do i5=1,size(sum_double_st_dim_6D,5)
          do i4=1,size(sum_double_st_dim_6D,4)
            do i3=1,size(sum_double_st_dim_6D,3)
              do i2=1,size(sum_double_st_dim_6D,2)
                do i1=1,size(sum_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    sum_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_6D

  function product_single_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:), allocatable :: product_single_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( product_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,6)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5)) )
        do i5=1,size(product_single_st_dim_6D,5)
          do i4=1,size(product_single_st_dim_6D,4)
            do i3=1,size(product_single_st_dim_6D,3)
              do i2=1,size(product_single_st_dim_6D,2)
                do i1=1,size(product_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    product_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_6D

  function product_double_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:), allocatable :: product_double_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( product_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,5), size(array,6)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,6)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5)) )
        do i5=1,size(product_double_st_dim_6D,5)
          do i4=1,size(product_double_st_dim_6D,4)
            do i3=1,size(product_double_st_dim_6D,3)
              do i2=1,size(product_double_st_dim_6D,2)
                do i1=1,size(product_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    product_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_6D

  function minloc_single_st_dim_6D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                         intent(in) :: back
    integer, dimension(:,:,:,:,:), allocatable :: minloc_single_st_dim_6D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5), back)
                  else
                    output = minloc(array(:,i1,i2,i3,i4,i5), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5), back)
                  else
                    output = minloc(array(i1,:,i2,i3,i4,i5), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5), back)
                  else
                    output = minloc(array(i1,i2,:,i3,i4,i5), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5), back)
                  else
                    output = minloc(array(i1,i2,i3,:,i4,i5), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5), back)
                  else
                    output = minloc(array(i1,i2,i3,i4,:,i5), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(minloc_single_st_dim_6D,5)
          do i4=1,size(minloc_single_st_dim_6D,4)
            do i3=1,size(minloc_single_st_dim_6D,3)
              do i2=1,size(minloc_single_st_dim_6D,2)
                do i1=1,size(minloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:), back)
                  else
                    output = minloc(array(i1,i2,i3,i4,i5,:), back=back)
                  end if
                  minloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_6D

  function minloc_double_st_dim_6D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                         intent(in) :: back
    integer, dimension(:,:,:,:,:), allocatable :: minloc_double_st_dim_6D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5), back)
                  else
                    output = minloc(array(:,i1,i2,i3,i4,i5), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5), back)
                  else
                    output = minloc(array(i1,:,i2,i3,i4,i5), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5), back)
                  else
                    output = minloc(array(i1,i2,:,i3,i4,i5), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5), back)
                  else
                    output = minloc(array(i1,i2,i3,:,i4,i5), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5), back)
                  else
                    output = minloc(array(i1,i2,i3,i4,:,i5), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(minloc_double_st_dim_6D,5)
          do i4=1,size(minloc_double_st_dim_6D,4)
            do i3=1,size(minloc_double_st_dim_6D,3)
              do i2=1,size(minloc_double_st_dim_6D,2)
                do i1=1,size(minloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = minloc(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:), back)
                  else
                    output = minloc(array(i1,i2,i3,i4,i5,:), back=back)
                  end if
                  minloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_6D

  function maxloc_single_st_dim_6D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                         intent(in) :: back
    integer, dimension(:,:,:,:,:), allocatable :: maxloc_single_st_dim_6D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5), back)
                  else
                    output = maxloc(array(:,i1,i2,i3,i4,i5), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5), back)
                  else
                    output = maxloc(array(i1,:,i2,i3,i4,i5), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5), back)
                  else
                    output = maxloc(array(i1,i2,:,i3,i4,i5), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5), back)
                  else
                    output = maxloc(array(i1,i2,i3,:,i4,i5), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5), back)
                  else
                    output = maxloc(array(i1,i2,i3,i4,:,i5), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxloc_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(maxloc_single_st_dim_6D,5)
          do i4=1,size(maxloc_single_st_dim_6D,4)
            do i3=1,size(maxloc_single_st_dim_6D,3)
              do i2=1,size(maxloc_single_st_dim_6D,2)
                do i1=1,size(maxloc_single_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:), back)
                  else
                    output = maxloc(array(i1,i2,i3,i4,i5,:), back=back)
                  end if
                  maxloc_single_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_6D

  function maxloc_double_st_dim_6D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                         intent(in) :: back
    integer, dimension(:,:,:,:,:), allocatable :: maxloc_double_st_dim_6D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5), back)
                  else
                    output = maxloc(array(:,i1,i2,i3,i4,i5), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5), back)
                  else
                    output = maxloc(array(i1,:,i2,i3,i4,i5), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5), back)
                  else
                    output = maxloc(array(i1,i2,:,i3,i4,i5), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5), back)
                  else
                    output = maxloc(array(i1,i2,i3,:,i4,i5), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5), back)
                  else
                    output = maxloc(array(i1,i2,i3,i4,:,i5), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxloc_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(maxloc_double_st_dim_6D,5)
          do i4=1,size(maxloc_double_st_dim_6D,4)
            do i3=1,size(maxloc_double_st_dim_6D,3)
              do i2=1,size(maxloc_double_st_dim_6D,2)
                do i1=1,size(maxloc_double_st_dim_6D,1)
                  if (present(mask)) then
                    output = maxloc(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:), back)
                  else
                    output = maxloc(array(i1,i2,i3,i4,i5,:), back=back)
                  end if
                  maxloc_double_st_dim_6D(i1,i2,i3,i4,i5) = output(1) 
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_6D

  function minval_single_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:), allocatable :: minval_single_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(minval_single_st_dim_6D,5)
          do i4=1,size(minval_single_st_dim_6D,4)
            do i3=1,size(minval_single_st_dim_6D,3)
              do i2=1,size(minval_single_st_dim_6D,2)
                do i1=1,size(minval_single_st_dim_6D,1)
                  if (present(mask)) then
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    minval_single_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minval_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_6D

  function minval_double_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:), allocatable :: minval_double_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(minval_double_st_dim_6D,5)
          do i4=1,size(minval_double_st_dim_6D,4)
            do i3=1,size(minval_double_st_dim_6D,3)
              do i2=1,size(minval_double_st_dim_6D,2)
                do i1=1,size(minval_double_st_dim_6D,1)
                  if (present(mask)) then
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    minval_double_st_dim_6D(i1,i2,i3,i4,i5) = minval(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minval_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_6D

  function maxval_single_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:), allocatable :: maxval_single_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_single_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_single_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxval_single_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(maxval_single_st_dim_6D,5)
          do i4=1,size(maxval_single_st_dim_6D,4)
            do i3=1,size(maxval_single_st_dim_6D,3)
              do i2=1,size(maxval_single_st_dim_6D,2)
                do i1=1,size(maxval_single_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    maxval_single_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_6D

  function maxval_double_st_dim_6D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:), intent(in) :: array
    integer,                                   intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:), allocatable :: maxval_double_st_dim_6D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_6D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(:,i1,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_double_st_dim_6D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,:,i2,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_double_st_dim_6D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,:,i3,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,:,i4,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,:,i5))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxval_double_st_dim_6D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5)) )
        do i5=1,size(maxval_double_st_dim_6D,5)
          do i4=1,size(maxval_double_st_dim_6D,4)
            do i3=1,size(maxval_double_st_dim_6D,3)
              do i2=1,size(maxval_double_st_dim_6D,2)
                do i1=1,size(maxval_double_st_dim_6D,1)
                  if (present(mask)) then
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    maxval_double_st_dim_6D(i1,i2,i3,i4,i5) = maxval(array(i1,i2,i3,i4,i5,:))
                  end if
                  
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_6D




        !`$1_$2_dim_7D = '`ifelse(`$1', `maxval', `format(`TINY(%s)', `ifelse(`$2', `single_st', `0.', `0.d0')')', `$1', `minval', `format(`HUGE(%s)', `ifelse(`$2', `single_st', `0.', `0.d0')')', `$1', `product', `ifelse(`$2', `single_st', `1.', `1.d0')', `$1', `sum', `ifelse(`$2', `single_st', `0.', `0.d0')', `0')'

  function sum_single_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:,:), allocatable :: sum_single_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( sum_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( sum_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(sum_single_st_dim_7D,6)
          do i5=1,size(sum_single_st_dim_7D,5)
            do i4=1,size(sum_single_st_dim_7D,4)
              do i3=1,size(sum_single_st_dim_7D,3)
                do i2=1,size(sum_single_st_dim_7D,2)
                  do i1=1,size(sum_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      sum_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_single_st_dim_7D

  function sum_double_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:,:), allocatable :: sum_double_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( sum_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( sum_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                       size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(sum_double_st_dim_7D,6)
          do i5=1,size(sum_double_st_dim_7D,5)
            do i4=1,size(sum_double_st_dim_7D,4)
              do i3=1,size(sum_double_st_dim_7D,3)
                do i2=1,size(sum_double_st_dim_7D,2)
                  do i1=1,size(sum_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      sum_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_double_st_dim_7D

  function product_single_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:,:), allocatable :: product_single_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( product_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( product_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(product_single_st_dim_7D,6)
          do i5=1,size(product_single_st_dim_7D,5)
            do i4=1,size(product_single_st_dim_7D,4)
              do i3=1,size(product_single_st_dim_7D,3)
                do i2=1,size(product_single_st_dim_7D,2)
                  do i1=1,size(product_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      product_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_single_st_dim_7D

  function product_double_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:,:), allocatable :: product_double_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( product_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( product_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                           size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(product_double_st_dim_7D,6)
          do i5=1,size(product_double_st_dim_7D,5)
            do i4=1,size(product_double_st_dim_7D,4)
              do i3=1,size(product_double_st_dim_7D,3)
                do i2=1,size(product_double_st_dim_7D,2)
                  do i1=1,size(product_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      product_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_double_st_dim_7D

  function minloc_single_st_dim_7D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                           intent(in) :: back
    integer, dimension(:,:,:,:,:,:), allocatable :: minloc_single_st_dim_7D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( minloc_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(:,i1,i2,i3,i4,i5,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,:,i2,i3,i4,i5,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,:,i3,i4,i5,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,:,i4,i5,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,:,i5,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,i5,:,i6), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( minloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(minloc_single_st_dim_7D,6)
          do i5=1,size(minloc_single_st_dim_7D,5)
            do i4=1,size(minloc_single_st_dim_7D,4)
              do i3=1,size(minloc_single_st_dim_7D,3)
                do i2=1,size(minloc_single_st_dim_7D,2)
                  do i1=1,size(minloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,i5,i6,:), back=back)
                    end if
                    minloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_single_st_dim_7D

  function minloc_double_st_dim_7D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                           intent(in) :: back
    integer, dimension(:,:,:,:,:,:), allocatable :: minloc_double_st_dim_7D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( minloc_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(:,i1,i2,i3,i4,i5,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,:,i2,i3,i4,i5,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,:,i3,i4,i5,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,:,i4,i5,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,:,i5,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,i5,:,i6), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( minloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(minloc_double_st_dim_7D,6)
          do i5=1,size(minloc_double_st_dim_7D,5)
            do i4=1,size(minloc_double_st_dim_7D,4)
              do i3=1,size(minloc_double_st_dim_7D,3)
                do i2=1,size(minloc_double_st_dim_7D,2)
                  do i1=1,size(minloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = minloc(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:), back)
                    else
                      output = minloc(array(i1,i2,i3,i4,i5,i6,:), back=back)
                    end if
                    minloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minloc_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minloc_double_st_dim_7D

  function minval_single_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:,:), allocatable :: minval_single_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( minval_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( minval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(minval_single_st_dim_7D,6)
          do i5=1,size(minval_single_st_dim_7D,5)
            do i4=1,size(minval_single_st_dim_7D,4)
              do i3=1,size(minval_single_st_dim_7D,3)
                do i2=1,size(minval_single_st_dim_7D,2)
                  do i1=1,size(minval_single_st_dim_7D,1)
                    if (present(mask)) then
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      minval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minval_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_single_st_dim_7D

  function minval_double_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:,:), allocatable :: minval_double_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( minval_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( minval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(minval_double_st_dim_7D,6)
          do i5=1,size(minval_double_st_dim_7D,5)
            do i4=1,size(minval_double_st_dim_7D,4)
              do i3=1,size(minval_double_st_dim_7D,3)
                do i2=1,size(minval_double_st_dim_7D,2)
                  do i1=1,size(minval_double_st_dim_7D,1)
                    if (present(mask)) then
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      minval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          minval(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( minval_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function minval_double_st_dim_7D

  function maxloc_single_st_dim_7D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                           intent(in) :: back
    integer, dimension(:,:,:,:,:,:), allocatable :: maxloc_single_st_dim_7D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( maxloc_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(:,i1,i2,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,:,i2,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,:,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,:,i4,i5,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,:,i5,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,i5,:,i6), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( maxloc_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(maxloc_single_st_dim_7D,6)
          do i5=1,size(maxloc_single_st_dim_7D,5)
            do i4=1,size(maxloc_single_st_dim_7D,4)
              do i3=1,size(maxloc_single_st_dim_7D,3)
                do i2=1,size(maxloc_single_st_dim_7D,2)
                  do i1=1,size(maxloc_single_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,i5,i6,:), back=back)
                    end if
                    maxloc_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_single_st_dim_7D

  function maxloc_double_st_dim_7D(array, dim, mask , back)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    logical, optional,                           intent(in) :: back
    integer, dimension(:,:,:,:,:,:), allocatable :: maxloc_double_st_dim_7D
    ! Local declaration
    integer, dimension(1) :: output 
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( maxloc_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(:,i1,i2,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,:,i2,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,:,i3,i4,i5,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,:,i4,i5,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,:,i5,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,i5,:,i6), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( maxloc_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(maxloc_double_st_dim_7D,6)
          do i5=1,size(maxloc_double_st_dim_7D,5)
            do i4=1,size(maxloc_double_st_dim_7D,4)
              do i3=1,size(maxloc_double_st_dim_7D,3)
                do i2=1,size(maxloc_double_st_dim_7D,2)
                  do i1=1,size(maxloc_double_st_dim_7D,1)
                    if (present(mask)) then
                      output = maxloc(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:), back)
                    else
                      output = maxloc(array(i1,i2,i3,i4,i5,i6,:), back=back)
                    end if
                    maxloc_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = output(1) 
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxloc_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxloc_double_st_dim_7D

  function maxval_single_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(single_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(single_st), dimension(:,:,:,:,:,:), allocatable :: maxval_single_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( maxval_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( maxval_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(maxval_single_st_dim_7D,6)
          do i5=1,size(maxval_single_st_dim_7D,5)
            do i4=1,size(maxval_single_st_dim_7D,4)
              do i3=1,size(maxval_single_st_dim_7D,3)
                do i2=1,size(maxval_single_st_dim_7D,2)
                  do i1=1,size(maxval_single_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      maxval_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_single_st_dim_7D

  function maxval_double_st_dim_7D(array, dim, mask )
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(double_st),   dimension(:,:,:,:,:,:,:), intent(in) :: array
    integer,                                     intent(in) :: dim
    logical, optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    
    type(double_st), dimension(:,:,:,:,:,:), allocatable :: maxval_double_st_dim_7D
    ! Local declaration
    
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( maxval_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,5), size(array,6), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,6), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,7)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( maxval_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                          size(array,4), size(array,5), size(array,6)) )
        do i6=1,size(maxval_double_st_dim_7D,6)
          do i5=1,size(maxval_double_st_dim_7D,5)
            do i4=1,size(maxval_double_st_dim_7D,4)
              do i3=1,size(maxval_double_st_dim_7D,3)
                do i2=1,size(maxval_double_st_dim_7D,2)
                  do i1=1,size(maxval_double_st_dim_7D,1)
                    if (present(mask)) then
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      maxval_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          maxval(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                    
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( maxval_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function maxval_double_st_dim_7D




end module cadna_array

