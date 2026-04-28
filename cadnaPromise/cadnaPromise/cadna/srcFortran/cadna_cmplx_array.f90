
! Module de surcharge des fonctions tableaux pour les complexes






module cadna_cmplx_array
  use cadna_types
  use cadna_cmplx_affect
  use cadna_cmplx_add
  use cadna_cmplx_mult
  use cadna_cmplx_convert

  implicit none




  interface dot_product
    module procedure dot_product_complex_single_st
  end interface dot_product
  interface dot_product
    module procedure dot_product_complex_double_st
  end interface dot_product



  interface matmul
    module procedure matmul_complex_single_st_1D2D
  end interface matmul
  interface matmul
    module procedure matmul_complex_double_st_1D2D
  end interface matmul
  interface matmul
    module procedure matmul_complex_single_st_2D1D
  end interface matmul
  interface matmul
    module procedure matmul_complex_double_st_2D1D
  end interface matmul
  interface matmul
    module procedure matmul_complex_single_st_2D2D
  end interface matmul
  interface matmul
    module procedure matmul_complex_double_st_2D2D
  end interface matmul



  interface sum
    module procedure sum_complex_single_st_1D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_1D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_1D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_1D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_2D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_2D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_2D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_2D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_3D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_3D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_3D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_3D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_4D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_4D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_4D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_4D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_5D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_5D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_5D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_5D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_6D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_6D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_6D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_6D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_7D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_7D
  end interface sum
  interface sum
    module procedure sum_complex_single_st_dim_7D
  end interface sum
  interface sum
    module procedure sum_complex_double_st_dim_7D
  end interface sum

  interface product
    module procedure product_complex_single_st_1D
  end interface product
  interface product
    module procedure product_complex_double_st_1D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_1D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_1D
  end interface product
  interface product
    module procedure product_complex_single_st_2D
  end interface product
  interface product
    module procedure product_complex_double_st_2D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_2D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_2D
  end interface product
  interface product
    module procedure product_complex_single_st_3D
  end interface product
  interface product
    module procedure product_complex_double_st_3D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_3D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_3D
  end interface product
  interface product
    module procedure product_complex_single_st_4D
  end interface product
  interface product
    module procedure product_complex_double_st_4D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_4D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_4D
  end interface product
  interface product
    module procedure product_complex_single_st_5D
  end interface product
  interface product
    module procedure product_complex_double_st_5D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_5D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_5D
  end interface product
  interface product
    module procedure product_complex_single_st_6D
  end interface product
  interface product
    module procedure product_complex_double_st_6D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_6D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_6D
  end interface product
  interface product
    module procedure product_complex_single_st_7D
  end interface product
  interface product
    module procedure product_complex_double_st_7D
  end interface product
  interface product
    module procedure product_complex_single_st_dim_7D
  end interface product
  interface product
    module procedure product_complex_double_st_dim_7D
  end interface product

contains



  function dot_product_complex_single_st(vector_a, vector_b)
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:), intent(in) :: vector_a, vector_b
    type(complex_single_st) :: dot_product_complex_single_st

    dot_product_complex_single_st = SUM(conjg(vector_a)*vector_b)
  end function dot_product_complex_single_st

  function dot_product_complex_double_st(vector_a, vector_b)
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:), intent(in) :: vector_a, vector_b
    type(complex_double_st) :: dot_product_complex_double_st

    dot_product_complex_double_st = SUM(conjg(vector_a)*vector_b)
  end function dot_product_complex_double_st




  function matmul_complex_single_st_1D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:), intent(in) :: matrix_a
    type(complex_single_st), dimension (:,:), intent(in) :: matrix_b 
    type(complex_single_st), dimension (size(matrix_b,2)) :: matmul_complex_single_st_1D2D
    ! Local declaration
    integer j

    do j=1,size(matrix_b,2)
      matmul_complex_single_st_1D2D(j) = SUM(matrix_a*matrix_b(:,j))
    enddo
  end function matmul_complex_single_st_1D2D

  function matmul_complex_double_st_1D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:), intent(in) :: matrix_a
    type(complex_double_st), dimension (:,:), intent(in) :: matrix_b 
    type(complex_double_st), dimension (size(matrix_b,2)) :: matmul_complex_double_st_1D2D
    ! Local declaration
    integer j

    do j=1,size(matrix_b,2)
      matmul_complex_double_st_1D2D(j) = SUM(matrix_a*matrix_b(:,j))
    enddo
  end function matmul_complex_double_st_1D2D

  function matmul_complex_single_st_2D1D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:), intent(in) :: matrix_a 
    type(complex_single_st), dimension (:), intent(in) :: matrix_b
    type(complex_single_st), dimension (size(matrix_a,1)) :: matmul_complex_single_st_2D1D
    ! Local declaration
    integer i

    do i=1,size(matrix_a,1)
      matmul_complex_single_st_2D1D(i) = SUM(matrix_a(i,:)*matrix_b)
    enddo
  end function matmul_complex_single_st_2D1D

  function matmul_complex_double_st_2D1D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:), intent(in) :: matrix_a 
    type(complex_double_st), dimension (:), intent(in) :: matrix_b
    type(complex_double_st), dimension (size(matrix_a,1)) :: matmul_complex_double_st_2D1D
    ! Local declaration
    integer i

    do i=1,size(matrix_a,1)
      matmul_complex_double_st_2D1D(i) = SUM(matrix_a(i,:)*matrix_b)
    enddo
  end function matmul_complex_double_st_2D1D

  function matmul_complex_single_st_2D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:), intent(in) :: matrix_a 
    type(complex_single_st), dimension (:,:), intent(in) :: matrix_b 
    type(complex_single_st), dimension (size(matrix_a,1), size(matrix_b,2)) :: matmul_complex_single_st_2D2D 
    ! Local declaration
    integer i, j 

    do i=1,size(matrix_a,1)
      do j=1,size(matrix_b,2)
        matmul_complex_single_st_2D2D(i,j) = SUM(matrix_a(i,:)*matrix_b(:,j))
      enddo
    enddo 
  end function matmul_complex_single_st_2D2D

  function matmul_complex_double_st_2D2D(matrix_a, matrix_b)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:), intent(in) :: matrix_a 
    type(complex_double_st), dimension (:,:), intent(in) :: matrix_b 
    type(complex_double_st), dimension (size(matrix_a,1), size(matrix_b,2)) :: matmul_complex_double_st_2D2D 
    ! Local declaration
    integer i, j 

    do i=1,size(matrix_a,1)
      do j=1,size(matrix_b,2)
        matmul_complex_double_st_2D2D(i,j) = SUM(matrix_a(i,:)*matrix_b(:,j))
      enddo
    enddo 
  end function matmul_complex_double_st_2D2D




  function sum_complex_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:), intent(in), target, contiguous :: array
    logical,                 dimension (:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_1D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_1D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_1D = sum_complex_single_st_1D + p_array(i)
    end do

  end function sum_complex_single_st_1D

  function sum_complex_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:), intent(in), target, contiguous :: array
    logical,                 dimension (:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_1D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_1D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_1D = sum_complex_double_st_1D + p_array(i)
    end do

  end function sum_complex_double_st_1D

  function sum_complex_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_2D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_2D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_2D = sum_complex_single_st_2D + p_array(i)
    end do

  end function sum_complex_single_st_2D

  function sum_complex_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_2D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_2D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_2D = sum_complex_double_st_2D + p_array(i)
    end do

  end function sum_complex_double_st_2D

  function sum_complex_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_3D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_3D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_3D = sum_complex_single_st_3D + p_array(i)
    end do

  end function sum_complex_single_st_3D

  function sum_complex_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_3D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_3D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_3D = sum_complex_double_st_3D + p_array(i)
    end do

  end function sum_complex_double_st_3D

  function sum_complex_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_4D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_4D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_4D = sum_complex_single_st_4D + p_array(i)
    end do

  end function sum_complex_single_st_4D

  function sum_complex_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_4D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_4D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_4D = sum_complex_double_st_4D + p_array(i)
    end do

  end function sum_complex_double_st_4D

  function sum_complex_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_5D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_5D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_5D = sum_complex_single_st_5D + p_array(i)
    end do

  end function sum_complex_single_st_5D

  function sum_complex_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_5D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_5D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_5D = sum_complex_double_st_5D + p_array(i)
    end do

  end function sum_complex_double_st_5D

  function sum_complex_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_6D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_6D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_6D = sum_complex_single_st_6D + p_array(i)
    end do

  end function sum_complex_single_st_6D

  function sum_complex_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_6D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_6D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_6D = sum_complex_double_st_6D + p_array(i)
    end do

  end function sum_complex_double_st_6D

  function sum_complex_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: sum_complex_single_st_7D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_single_st_7D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_single_st_7D = sum_complex_single_st_7D + p_array(i)
    end do

  end function sum_complex_single_st_7D

  function sum_complex_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: sum_complex_double_st_7D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    sum_complex_double_st_7D = 0
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      sum_complex_double_st_7D = sum_complex_double_st_7D + p_array(i)
    end do

  end function sum_complex_double_st_7D


  function product_complex_single_st_1D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:), intent(in), target, contiguous :: array
    logical,                 dimension (:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_1D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_1D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_1D = product_complex_single_st_1D * p_array(i) 
    end do

  end function product_complex_single_st_1D

  function product_complex_double_st_1D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:), intent(in), target, contiguous :: array
    logical,                 dimension (:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_1D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_1D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_1D = product_complex_double_st_1D * p_array(i) 
    end do

  end function product_complex_double_st_1D

  function product_complex_single_st_2D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_2D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_2D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_2D = product_complex_single_st_2D * p_array(i) 
    end do

  end function product_complex_single_st_2D

  function product_complex_double_st_2D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_2D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_2D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_2D = product_complex_double_st_2D * p_array(i) 
    end do

  end function product_complex_double_st_2D

  function product_complex_single_st_3D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_3D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_3D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_3D = product_complex_single_st_3D * p_array(i) 
    end do

  end function product_complex_single_st_3D

  function product_complex_double_st_3D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_3D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_3D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_3D = product_complex_double_st_3D * p_array(i) 
    end do

  end function product_complex_double_st_3D

  function product_complex_single_st_4D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_4D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_4D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_4D = product_complex_single_st_4D * p_array(i) 
    end do

  end function product_complex_single_st_4D

  function product_complex_double_st_4D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_4D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_4D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_4D = product_complex_double_st_4D * p_array(i) 
    end do

  end function product_complex_double_st_4D

  function product_complex_single_st_5D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_5D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_5D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_5D = product_complex_single_st_5D * p_array(i) 
    end do

  end function product_complex_single_st_5D

  function product_complex_double_st_5D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_5D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_5D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_5D = product_complex_double_st_5D * p_array(i) 
    end do

  end function product_complex_double_st_5D

  function product_complex_single_st_6D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_6D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_6D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_6D = product_complex_single_st_6D * p_array(i) 
    end do

  end function product_complex_single_st_6D

  function product_complex_double_st_6D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_6D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_6D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_6D = product_complex_double_st_6D * p_array(i) 
    end do

  end function product_complex_double_st_6D

  function product_complex_single_st_7D(array, mask)
    ! Dummy arguments declaration
    type(complex_single_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_single_st) :: product_complex_single_st_7D
    ! Local declaration
    type(complex_single_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_single_st_7D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_single_st_7D = product_complex_single_st_7D * p_array(i) 
    end do

  end function product_complex_single_st_7D

  function product_complex_double_st_7D(array, mask)
    ! Dummy arguments declaration
    type(complex_double_st), dimension (:,:,:,:,:,:,:), intent(in), target, contiguous :: array
    logical,                 dimension (:,:,:,:,:,:,:), intent(in), optional, target, contiguous :: mask
    type(complex_double_st) :: product_complex_double_st_7D
    ! Local declaration
    type(complex_double_st), dimension(:), pointer :: p_array
    logical,                 dimension(:), pointer :: p_mask
    integer i

    p_array(1:size(array)) => array
    if (present(mask)) p_mask(1:size(mask)) => mask
    product_complex_double_st_7D = 1
    do i=1,size(p_array)
      if (present(mask)) then
        if( .not.p_mask(i) ) cycle
      end if
      product_complex_double_st_7D = product_complex_double_st_7D * p_array(i) 
    end do

  end function product_complex_double_st_7D




  function sum_complex_single_st_dim_1D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:), intent(in) :: array
    logical,       optional, dimension(:), intent(in) :: mask
    integer,                               intent(in) :: dim
    type(complex_single_st) :: sum_complex_single_st_dim_1D

    select case(dim)
      case (1)
        if (present(mask)) then
          sum_complex_single_st_dim_1D = sum(array, mask)
        else
          sum_complex_single_st_dim_1D = sum(array)
        end if
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        sum_complex_single_st_dim_1D = 0.
    end select
  end function sum_complex_single_st_dim_1D

  function sum_complex_double_st_dim_1D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:), intent(in) :: array
    logical,       optional, dimension(:), intent(in) :: mask
    integer,                               intent(in) :: dim
    type(complex_double_st) :: sum_complex_double_st_dim_1D

    select case(dim)
      case (1)
        if (present(mask)) then
          sum_complex_double_st_dim_1D = sum(array, mask)
        else
          sum_complex_double_st_dim_1D = sum(array)
        end if
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        sum_complex_double_st_dim_1D = 0.d0
    end select
  end function sum_complex_double_st_dim_1D

  function product_complex_single_st_dim_1D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:), intent(in) :: array
    logical,       optional, dimension(:), intent(in) :: mask
    integer,                               intent(in) :: dim
    type(complex_single_st) :: product_complex_single_st_dim_1D

    select case(dim)
      case (1)
        if (present(mask)) then
          product_complex_single_st_dim_1D = product(array, mask)
        else
          product_complex_single_st_dim_1D = product(array)
        end if
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        product_complex_single_st_dim_1D = 1. 
    end select
  end function product_complex_single_st_dim_1D

  function product_complex_double_st_dim_1D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:), intent(in) :: array
    logical,       optional, dimension(:), intent(in) :: mask
    integer,                               intent(in) :: dim
    type(complex_double_st) :: product_complex_double_st_dim_1D

    select case(dim)
      case (1)
        if (present(mask)) then
          product_complex_double_st_dim_1D = product(array, mask)
        else
          product_complex_double_st_dim_1D = product(array)
        end if
      case default
        write(ERROR_UNIT, *) "Incorrect value of DIM argument."
        product_complex_double_st_dim_1D = 1.d0 
    end select
  end function product_complex_double_st_dim_1D




  function sum_complex_single_st_dim_2D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:), intent(in) :: array
    logical,       optional, dimension(:,:), intent(in) :: mask
    integer,                                 intent(in) :: dim
    type(complex_single_st), dimension(:), allocatable :: sum_complex_single_st_dim_2D
    ! Local declaration
    integer i1

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(sum_complex_single_st_dim_2D,1)
          if (present(mask)) then
            sum_complex_single_st_dim_2D(i1) = sum(array(:,i1), mask(:,i1))
          else
            sum_complex_single_st_dim_2D(i1) = sum(array(:,i1))
          end if
        end do
      case (2)
        allocate( sum_complex_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(sum_complex_single_st_dim_2D,1)
          if (present(mask)) then
            sum_complex_single_st_dim_2D(i1) = sum(array(i1,:), mask(i1,:))
          else
            sum_complex_single_st_dim_2D(i1) = sum(array(i1,:))
          end if
        end do
      case default
        allocate( sum_complex_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_2D

  function sum_complex_double_st_dim_2D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:), intent(in) :: array
    logical,       optional, dimension(:,:), intent(in) :: mask
    integer,                                 intent(in) :: dim
    type(complex_double_st), dimension(:), allocatable :: sum_complex_double_st_dim_2D
    ! Local declaration
    integer i1

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(sum_complex_double_st_dim_2D,1)
          if (present(mask)) then
            sum_complex_double_st_dim_2D(i1) = sum(array(:,i1), mask(:,i1))
          else
            sum_complex_double_st_dim_2D(i1) = sum(array(:,i1))
          end if
        end do
      case (2)
        allocate( sum_complex_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(sum_complex_double_st_dim_2D,1)
          if (present(mask)) then
            sum_complex_double_st_dim_2D(i1) = sum(array(i1,:), mask(i1,:))
          else
            sum_complex_double_st_dim_2D(i1) = sum(array(i1,:))
          end if
        end do
      case default
        allocate( sum_complex_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_2D

  function product_complex_single_st_dim_2D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:), intent(in) :: array
    logical,       optional, dimension(:,:), intent(in) :: mask
    integer,                                 intent(in) :: dim
    type(complex_single_st), dimension(:), allocatable :: product_complex_single_st_dim_2D
    ! Local declaration
    integer i1

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_2D(size(array,2)) ) 
        do i1=1,size(product_complex_single_st_dim_2D,1)
          if (present(mask)) then
            product_complex_single_st_dim_2D(i1) = product(array(:,i1), mask(:,i1))
          else
            product_complex_single_st_dim_2D(i1) = product(array(:,i1))
          end if
        end do
      case (2)
        allocate( product_complex_single_st_dim_2D(size(array,1)) ) 
        do i1=1,size(product_complex_single_st_dim_2D,1)
          if (present(mask)) then
            product_complex_single_st_dim_2D(i1) = product(array(i1,:), mask(i1,:))
          else
            product_complex_single_st_dim_2D(i1) = product(array(i1,:))
          end if
        end do
      case default
        allocate( product_complex_single_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_2D

  function product_complex_double_st_dim_2D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:), intent(in) :: array
    logical,       optional, dimension(:,:), intent(in) :: mask
    integer,                                 intent(in) :: dim
    type(complex_double_st), dimension(:), allocatable :: product_complex_double_st_dim_2D
    ! Local declaration
    integer i1

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_2D(size(array,2)) ) 
        do i1=1,size(product_complex_double_st_dim_2D,1)
          if (present(mask)) then
            product_complex_double_st_dim_2D(i1) = product(array(:,i1), mask(:,i1))
          else
            product_complex_double_st_dim_2D(i1) = product(array(:,i1))
          end if
        end do
      case (2)
        allocate( product_complex_double_st_dim_2D(size(array,1)) ) 
        do i1=1,size(product_complex_double_st_dim_2D,1)
          if (present(mask)) then
            product_complex_double_st_dim_2D(i1) = product(array(i1,:), mask(i1,:))
          else
            product_complex_double_st_dim_2D(i1) = product(array(i1,:))
          end if
        end do
      case default
        allocate( product_complex_double_st_dim_2D(0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_2D




  function sum_complex_single_st_dim_3D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:), intent(in) :: mask
    integer,                                   intent(in) :: dim
    type(complex_single_st), dimension(:,:), allocatable :: sum_complex_single_st_dim_3D
    ! Local declaration
    integer i1, i2

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(sum_complex_single_st_dim_3D,2)
          do i1=1,size(sum_complex_single_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(:,i1,i2), mask(:,i1,i2))
            else
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(:,i1,i2))
            end if
          end do
        end do
      case (2)
        allocate( sum_complex_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(sum_complex_single_st_dim_3D,2)
          do i1=1,size(sum_complex_single_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(i1,:,i2), mask(i1,:,i2))
            else
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(i1,:,i2))
            end if
          end do
        end do
      case (3)
        allocate( sum_complex_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(sum_complex_single_st_dim_3D,2)
          do i1=1,size(sum_complex_single_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(i1,i2,:), mask(i1,i2,:))
            else
              sum_complex_single_st_dim_3D(i1,i2) = sum(array(i1,i2,:))
            end if
          end do
        end do
      case default
        allocate( sum_complex_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_3D

  function sum_complex_double_st_dim_3D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:), intent(in) :: mask
    integer,                                   intent(in) :: dim
    type(complex_double_st), dimension(:,:), allocatable :: sum_complex_double_st_dim_3D
    ! Local declaration
    integer i1, i2

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(sum_complex_double_st_dim_3D,2)
          do i1=1,size(sum_complex_double_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(:,i1,i2), mask(:,i1,i2))
            else
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(:,i1,i2))
            end if
          end do
        end do
      case (2)
        allocate( sum_complex_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(sum_complex_double_st_dim_3D,2)
          do i1=1,size(sum_complex_double_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(i1,:,i2), mask(i1,:,i2))
            else
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(i1,:,i2))
            end if
          end do
        end do
      case (3)
        allocate( sum_complex_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(sum_complex_double_st_dim_3D,2)
          do i1=1,size(sum_complex_double_st_dim_3D,1)
            if (present(mask)) then
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(i1,i2,:), mask(i1,i2,:))
            else
              sum_complex_double_st_dim_3D(i1,i2) = sum(array(i1,i2,:))
            end if
          end do
        end do
      case default
        allocate( sum_complex_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_3D

  function product_complex_single_st_dim_3D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:), intent(in) :: mask
    integer,                                   intent(in) :: dim
    type(complex_single_st), dimension(:,:), allocatable :: product_complex_single_st_dim_3D
    ! Local declaration
    integer i1, i2

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(product_complex_single_st_dim_3D,2)
          do i1=1,size(product_complex_single_st_dim_3D,1)
            if (present(mask)) then
              product_complex_single_st_dim_3D(i1,i2) = product(array(:,i1,i2), mask(:,i1,i2))
            else
              product_complex_single_st_dim_3D(i1,i2) = product(array(:,i1,i2))
            end if
          end do
        end do
      case (2)
        allocate( product_complex_single_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(product_complex_single_st_dim_3D,2)
          do i1=1,size(product_complex_single_st_dim_3D,1)
            if (present(mask)) then
              product_complex_single_st_dim_3D(i1,i2) = product(array(i1,:,i2), mask(i1,:,i2))
            else
              product_complex_single_st_dim_3D(i1,i2) = product(array(i1,:,i2))
            end if
          end do
        end do
      case (3)
        allocate( product_complex_single_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(product_complex_single_st_dim_3D,2)
          do i1=1,size(product_complex_single_st_dim_3D,1)
            if (present(mask)) then
              product_complex_single_st_dim_3D(i1,i2) = product(array(i1,i2,:), mask(i1,i2,:))
            else
              product_complex_single_st_dim_3D(i1,i2) = product(array(i1,i2,:))
            end if
          end do
        end do
      case default
        allocate( product_complex_single_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_3D

  function product_complex_double_st_dim_3D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:), intent(in) :: mask
    integer,                                   intent(in) :: dim
    type(complex_double_st), dimension(:,:), allocatable :: product_complex_double_st_dim_3D
    ! Local declaration
    integer i1, i2

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_3D(size(array,2), size(array,3)) ) 
        do i2=1,size(product_complex_double_st_dim_3D,2)
          do i1=1,size(product_complex_double_st_dim_3D,1)
            if (present(mask)) then
              product_complex_double_st_dim_3D(i1,i2) = product(array(:,i1,i2), mask(:,i1,i2))
            else
              product_complex_double_st_dim_3D(i1,i2) = product(array(:,i1,i2))
            end if
          end do
        end do
      case (2)
        allocate( product_complex_double_st_dim_3D(size(array,1), size(array,3)) ) 
        do i2=1,size(product_complex_double_st_dim_3D,2)
          do i1=1,size(product_complex_double_st_dim_3D,1)
            if (present(mask)) then
              product_complex_double_st_dim_3D(i1,i2) = product(array(i1,:,i2), mask(i1,:,i2))
            else
              product_complex_double_st_dim_3D(i1,i2) = product(array(i1,:,i2))
            end if
          end do
        end do
      case (3)
        allocate( product_complex_double_st_dim_3D(size(array,1), size(array,2)) ) 
        do i2=1,size(product_complex_double_st_dim_3D,2)
          do i1=1,size(product_complex_double_st_dim_3D,1)
            if (present(mask)) then
              product_complex_double_st_dim_3D(i1,i2) = product(array(i1,i2,:), mask(i1,i2,:))
            else
              product_complex_double_st_dim_3D(i1,i2) = product(array(i1,i2,:))
            end if
          end do
        end do
      case default
        allocate( product_complex_double_st_dim_3D(0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_3D




  function sum_complex_single_st_dim_4D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:), intent(in) :: mask
    integer,                                     intent(in) :: dim
    type(complex_single_st), dimension(:,:,:), allocatable :: sum_complex_single_st_dim_4D
    ! Local declaration
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_complex_single_st_dim_4D,3)
          do i2=1,size(sum_complex_single_st_dim_4D,2)
            do i1=1,size(sum_complex_single_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3))
              end if
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_complex_single_st_dim_4D,3)
          do i2=1,size(sum_complex_single_st_dim_4D,2)
            do i1=1,size(sum_complex_single_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3))
              end if
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(sum_complex_single_st_dim_4D,3)
          do i2=1,size(sum_complex_single_st_dim_4D,2)
            do i1=1,size(sum_complex_single_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3))
              end if
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(sum_complex_single_st_dim_4D,3)
          do i2=1,size(sum_complex_single_st_dim_4D,2)
            do i1=1,size(sum_complex_single_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                sum_complex_single_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:))
              end if
            end do
          end do
        end do
      case default
        allocate( sum_complex_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_4D

  function sum_complex_double_st_dim_4D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:), intent(in) :: mask
    integer,                                     intent(in) :: dim
    type(complex_double_st), dimension(:,:,:), allocatable :: sum_complex_double_st_dim_4D
    ! Local declaration
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_complex_double_st_dim_4D,3)
          do i2=1,size(sum_complex_double_st_dim_4D,2)
            do i1=1,size(sum_complex_double_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(:,i1,i2,i3))
              end if
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(sum_complex_double_st_dim_4D,3)
          do i2=1,size(sum_complex_double_st_dim_4D,2)
            do i1=1,size(sum_complex_double_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,:,i2,i3))
              end if
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(sum_complex_double_st_dim_4D,3)
          do i2=1,size(sum_complex_double_st_dim_4D,2)
            do i1=1,size(sum_complex_double_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,:,i3))
              end if
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(sum_complex_double_st_dim_4D,3)
          do i2=1,size(sum_complex_double_st_dim_4D,2)
            do i1=1,size(sum_complex_double_st_dim_4D,1)
              if (present(mask)) then
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                sum_complex_double_st_dim_4D(i1,i2,i3) = sum(array(i1,i2,i3,:))
              end if
            end do
          end do
        end do
      case default
        allocate( sum_complex_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_4D

  function product_complex_single_st_dim_4D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:), intent(in) :: mask
    integer,                                     intent(in) :: dim
    type(complex_single_st), dimension(:,:,:), allocatable :: product_complex_single_st_dim_4D
    ! Local declaration
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(product_complex_single_st_dim_4D,3)
          do i2=1,size(product_complex_single_st_dim_4D,2)
            do i1=1,size(product_complex_single_st_dim_4D,1)
              if (present(mask)) then
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3))
              end if
            end do
          end do
        end do
      case (2)
        allocate( product_complex_single_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(product_complex_single_st_dim_4D,3)
          do i2=1,size(product_complex_single_st_dim_4D,2)
            do i1=1,size(product_complex_single_st_dim_4D,1)
              if (present(mask)) then
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3))
              end if
            end do
          end do
        end do
      case (3)
        allocate( product_complex_single_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(product_complex_single_st_dim_4D,3)
          do i2=1,size(product_complex_single_st_dim_4D,2)
            do i1=1,size(product_complex_single_st_dim_4D,1)
              if (present(mask)) then
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3))
              end if
            end do
          end do
        end do
      case (4)
        allocate( product_complex_single_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(product_complex_single_st_dim_4D,3)
          do i2=1,size(product_complex_single_st_dim_4D,2)
            do i1=1,size(product_complex_single_st_dim_4D,1)
              if (present(mask)) then
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                product_complex_single_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:))
              end if
            end do
          end do
        end do
      case default
        allocate( product_complex_single_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_4D

  function product_complex_double_st_dim_4D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:), intent(in) :: mask
    integer,                                     intent(in) :: dim
    type(complex_double_st), dimension(:,:,:), allocatable :: product_complex_double_st_dim_4D
    ! Local declaration
    integer i1, i2, i3

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_4D(size(array,2), size(array,3), size(array,4)) ) 
        do i3=1,size(product_complex_double_st_dim_4D,3)
          do i2=1,size(product_complex_double_st_dim_4D,2)
            do i1=1,size(product_complex_double_st_dim_4D,1)
              if (present(mask)) then
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3), mask(:,i1,i2,i3))
              else
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(:,i1,i2,i3))
              end if
            end do
          end do
        end do
      case (2)
        allocate( product_complex_double_st_dim_4D(size(array,1), size(array,3), size(array,4)) ) 
        do i3=1,size(product_complex_double_st_dim_4D,3)
          do i2=1,size(product_complex_double_st_dim_4D,2)
            do i1=1,size(product_complex_double_st_dim_4D,1)
              if (present(mask)) then
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3), mask(i1,:,i2,i3))
              else
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,:,i2,i3))
              end if
            end do
          end do
        end do
      case (3)
        allocate( product_complex_double_st_dim_4D(size(array,1), size(array,2), size(array,4)) ) 
        do i3=1,size(product_complex_double_st_dim_4D,3)
          do i2=1,size(product_complex_double_st_dim_4D,2)
            do i1=1,size(product_complex_double_st_dim_4D,1)
              if (present(mask)) then
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3), mask(i1,i2,:,i3))
              else
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,:,i3))
              end if
            end do
          end do
        end do
      case (4)
        allocate( product_complex_double_st_dim_4D(size(array,1), size(array,2), size(array,3)) ) 
        do i3=1,size(product_complex_double_st_dim_4D,3)
          do i2=1,size(product_complex_double_st_dim_4D,2)
            do i1=1,size(product_complex_double_st_dim_4D,1)
              if (present(mask)) then
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:), mask(i1,i2,i3,:))
              else
                product_complex_double_st_dim_4D(i1,i2,i3) = product(array(i1,i2,i3,:))
              end if
            end do
          end do
        end do
      case default
        allocate( product_complex_double_st_dim_4D(0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_4D




  function sum_complex_single_st_dim_5D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:), intent(in) :: mask
    integer,                                       intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:), allocatable :: sum_complex_single_st_dim_5D
    ! Local declaration
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_single_st_dim_5D,4)
          do i3=1,size(sum_complex_single_st_dim_5D,3)
            do i2=1,size(sum_complex_single_st_dim_5D,2)
              do i1=1,size(sum_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_single_st_dim_5D,4)
          do i3=1,size(sum_complex_single_st_dim_5D,3)
            do i2=1,size(sum_complex_single_st_dim_5D,2)
              do i1=1,size(sum_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_single_st_dim_5D,4)
          do i3=1,size(sum_complex_single_st_dim_5D,3)
            do i2=1,size(sum_complex_single_st_dim_5D,2)
              do i1=1,size(sum_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5) )) 
        do i4=1,size(sum_complex_single_st_dim_5D,4)
          do i3=1,size(sum_complex_single_st_dim_5D,3)
            do i2=1,size(sum_complex_single_st_dim_5D,2)
              do i1=1,size(sum_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4))
                end if
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(sum_complex_single_st_dim_5D,4)
          do i3=1,size(sum_complex_single_st_dim_5D,3)
            do i2=1,size(sum_complex_single_st_dim_5D,2)
              do i1=1,size(sum_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  sum_complex_single_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:))
                end if
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_5D

  function sum_complex_double_st_dim_5D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:), intent(in) :: mask
    integer,                                       intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:), allocatable :: sum_complex_double_st_dim_5D
    ! Local declaration
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_double_st_dim_5D,4)
          do i3=1,size(sum_complex_double_st_dim_5D,3)
            do i2=1,size(sum_complex_double_st_dim_5D,2)
              do i1=1,size(sum_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(:,i1,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_double_st_dim_5D,4)
          do i3=1,size(sum_complex_double_st_dim_5D,3)
            do i2=1,size(sum_complex_double_st_dim_5D,2)
              do i1=1,size(sum_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,:,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(sum_complex_double_st_dim_5D,4)
          do i3=1,size(sum_complex_double_st_dim_5D,3)
            do i2=1,size(sum_complex_double_st_dim_5D,2)
              do i1=1,size(sum_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,:,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5) )) 
        do i4=1,size(sum_complex_double_st_dim_5D,4)
          do i3=1,size(sum_complex_double_st_dim_5D,3)
            do i2=1,size(sum_complex_double_st_dim_5D,2)
              do i1=1,size(sum_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,:,i4))
                end if
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(sum_complex_double_st_dim_5D,4)
          do i3=1,size(sum_complex_double_st_dim_5D,3)
            do i2=1,size(sum_complex_double_st_dim_5D,2)
              do i1=1,size(sum_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  sum_complex_double_st_dim_5D(i1,i2,i3,i4) = sum(array(i1,i2,i3,i4,:))
                end if
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_5D

  function product_complex_single_st_dim_5D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:), intent(in) :: mask
    integer,                                       intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:), allocatable :: product_complex_single_st_dim_5D
    ! Local declaration
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_single_st_dim_5D,4)
          do i3=1,size(product_complex_single_st_dim_5D,3)
            do i2=1,size(product_complex_single_st_dim_5D,2)
              do i1=1,size(product_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_single_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_single_st_dim_5D,4)
          do i3=1,size(product_complex_single_st_dim_5D,3)
            do i2=1,size(product_complex_single_st_dim_5D,2)
              do i1=1,size(product_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_single_st_dim_5D,4)
          do i3=1,size(product_complex_single_st_dim_5D,3)
            do i2=1,size(product_complex_single_st_dim_5D,2)
              do i1=1,size(product_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5) )) 
        do i4=1,size(product_complex_single_st_dim_5D,4)
          do i3=1,size(product_complex_single_st_dim_5D,3)
            do i2=1,size(product_complex_single_st_dim_5D,2)
              do i1=1,size(product_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4))
                end if
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_single_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(product_complex_single_st_dim_5D,4)
          do i3=1,size(product_complex_single_st_dim_5D,3)
            do i2=1,size(product_complex_single_st_dim_5D,2)
              do i1=1,size(product_complex_single_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  product_complex_single_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:))
                end if
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_single_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_5D

  function product_complex_double_st_dim_5D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:), intent(in) :: mask
    integer,                                       intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:), allocatable :: product_complex_double_st_dim_5D
    ! Local declaration
    integer i1, i2, i3, i4

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_5D(size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_double_st_dim_5D,4)
          do i3=1,size(product_complex_double_st_dim_5D,3)
            do i2=1,size(product_complex_double_st_dim_5D,2)
              do i1=1,size(product_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4), mask(:,i1,i2,i3,i4))
                else
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(:,i1,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_double_st_dim_5D(size(array,1), size(array,3), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_double_st_dim_5D,4)
          do i3=1,size(product_complex_double_st_dim_5D,3)
            do i2=1,size(product_complex_double_st_dim_5D,2)
              do i1=1,size(product_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4), mask(i1,:,i2,i3,i4))
                else
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,:,i2,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,4), size(array,5)) ) 
        do i4=1,size(product_complex_double_st_dim_5D,4)
          do i3=1,size(product_complex_double_st_dim_5D,3)
            do i2=1,size(product_complex_double_st_dim_5D,2)
              do i1=1,size(product_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4), mask(i1,i2,:,i3,i4))
                else
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,:,i3,i4))
                end if
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,5) )) 
        do i4=1,size(product_complex_double_st_dim_5D,4)
          do i3=1,size(product_complex_double_st_dim_5D,3)
            do i2=1,size(product_complex_double_st_dim_5D,2)
              do i1=1,size(product_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4), mask(i1,i2,i3,:,i4))
                else
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,:,i4))
                end if
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_double_st_dim_5D(size(array,1), size(array,2), size(array,3), size(array,4)) ) 
        do i4=1,size(product_complex_double_st_dim_5D,4)
          do i3=1,size(product_complex_double_st_dim_5D,3)
            do i2=1,size(product_complex_double_st_dim_5D,2)
              do i1=1,size(product_complex_double_st_dim_5D,1)
                if (present(mask)) then
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:), mask(i1,i2,i3,i4,:))
                else
                  product_complex_double_st_dim_5D(i1,i2,i3,i4) = product(array(i1,i2,i3,i4,:))
                end if
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_double_st_dim_5D(0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_5D




  function sum_complex_single_st_dim_6D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    integer,                                         intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:,:), allocatable :: sum_complex_single_st_dim_6D
    ! Local declaration
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_6D(size(array,2), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_single_st_dim_6D(size(array,1), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,6)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i5=1,size(sum_complex_single_st_dim_6D,5)
          do i4=1,size(sum_complex_single_st_dim_6D,4)
            do i3=1,size(sum_complex_single_st_dim_6D,3)
              do i2=1,size(sum_complex_single_st_dim_6D,2)
                do i1=1,size(sum_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    sum_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:))
                  end if
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_6D

  function sum_complex_double_st_dim_6D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    integer,                                         intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:,:), allocatable :: sum_complex_double_st_dim_6D
    ! Local declaration
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_6D(size(array,2), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(:,i1,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_double_st_dim_6D(size(array,1), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,:,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,:,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,5), size(array,6)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,:,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,6)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,:,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i5=1,size(sum_complex_double_st_dim_6D,5)
          do i4=1,size(sum_complex_double_st_dim_6D,4)
            do i3=1,size(sum_complex_double_st_dim_6D,3)
              do i2=1,size(sum_complex_double_st_dim_6D,2)
                do i1=1,size(sum_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    sum_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = sum(array(i1,i2,i3,i4,i5,:))
                  end if
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_6D

  function product_complex_single_st_dim_6D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    integer,                                         intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:,:), allocatable :: product_complex_single_st_dim_6D
    ! Local declaration
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_6D(size(array,2), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_single_st_dim_6D(size(array,1), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,6)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_complex_single_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i5=1,size(product_complex_single_st_dim_6D,5)
          do i4=1,size(product_complex_single_st_dim_6D,4)
            do i3=1,size(product_complex_single_st_dim_6D,3)
              do i2=1,size(product_complex_single_st_dim_6D,2)
                do i1=1,size(product_complex_single_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    product_complex_single_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:))
                  end if
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_single_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_6D

  function product_complex_double_st_dim_6D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:), intent(in) :: mask
    integer,                                         intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:,:), allocatable :: product_complex_double_st_dim_6D
    ! Local declaration
    integer i1, i2, i3, i4, i5

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_6D(size(array,2), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5), mask(:,i1,i2,i3,i4,i5))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(:,i1,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_double_st_dim_6D(size(array,1), size(array,3), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5), mask(i1,:,i2,i3,i4,i5))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,:,i2,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,4), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5), mask(i1,i2,:,i3,i4,i5))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,:,i3,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,5), size(array,6)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5), mask(i1,i2,i3,:,i4,i5))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,:,i4,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,6)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5), mask(i1,i2,i3,i4,:,i5))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,:,i5))
                  end if
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_complex_double_st_dim_6D(size(array,1), size(array,2), size(array,3), size(array,4), size(array,5)) ) 
        do i5=1,size(product_complex_double_st_dim_6D,5)
          do i4=1,size(product_complex_double_st_dim_6D,4)
            do i3=1,size(product_complex_double_st_dim_6D,3)
              do i2=1,size(product_complex_double_st_dim_6D,2)
                do i1=1,size(product_complex_double_st_dim_6D,1)
                  if (present(mask)) then
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:), mask(i1,i2,i3,i4,i5,:))
                  else
                    product_complex_double_st_dim_6D(i1,i2,i3,i4,i5) = product(array(i1,i2,i3,i4,i5,:))
                  end if
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_double_st_dim_6D(0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_6D




  function sum_complex_single_st_dim_7D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    integer,                                           intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:,:,:), allocatable :: sum_complex_single_st_dim_7D
    ! Local declaration
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( sum_complex_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,7)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( sum_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,6)) ) 
        do i6=1,size(sum_complex_single_st_dim_7D,6)
          do i5=1,size(sum_complex_single_st_dim_7D,5)
            do i4=1,size(sum_complex_single_st_dim_7D,4)
              do i3=1,size(sum_complex_single_st_dim_7D,3)
                do i2=1,size(sum_complex_single_st_dim_7D,2)
                  do i1=1,size(sum_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      sum_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_single_st_dim_7D

  function sum_complex_double_st_dim_7D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    integer,                                           intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:,:,:), allocatable :: sum_complex_double_st_dim_7D
    ! Local declaration
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( sum_complex_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,6), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,7)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( sum_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,6)) ) 
        do i6=1,size(sum_complex_double_st_dim_7D,6)
          do i5=1,size(sum_complex_double_st_dim_7D,5)
            do i4=1,size(sum_complex_double_st_dim_7D,4)
              do i3=1,size(sum_complex_double_st_dim_7D,3)
                do i2=1,size(sum_complex_double_st_dim_7D,2)
                  do i1=1,size(sum_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      sum_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          sum(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( sum_complex_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function sum_complex_double_st_dim_7D

  function product_complex_single_st_dim_7D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_single_st), dimension(:,:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    integer,                                           intent(in) :: dim
    type(complex_single_st), dimension(:,:,:,:,:,:), allocatable :: product_complex_single_st_dim_7D
    ! Local declaration
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( product_complex_single_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,7)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( product_complex_single_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,6)) ) 
        do i6=1,size(product_complex_single_st_dim_7D,6)
          do i5=1,size(product_complex_single_st_dim_7D,5)
            do i4=1,size(product_complex_single_st_dim_7D,4)
              do i3=1,size(product_complex_single_st_dim_7D,3)
                do i2=1,size(product_complex_single_st_dim_7D,2)
                  do i1=1,size(product_complex_single_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      product_complex_single_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_single_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_single_st_dim_7D

  function product_complex_double_st_dim_7D(array, dim, mask)
    use ISO_FORTRAN_ENV, only : ERROR_UNIT
    ! Dummy arguments declaration
    type(complex_double_st), dimension(:,:,:,:,:,:,:), intent(in) :: array
    logical,       optional, dimension(:,:,:,:,:,:,:), intent(in) :: mask
    integer,                                           intent(in) :: dim
    type(complex_double_st), dimension(:,:,:,:,:,:), allocatable :: product_complex_double_st_dim_7D
    ! Local declaration
    integer i1, i2, i3, i4, i5, i6

    select case (dim)
      case (1)
        allocate( product_complex_double_st_dim_7D(size(array,2), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6), mask(:,i1,i2,i3,i4,i5,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(:,i1,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (2)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,3), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6), mask(i1,:,i2,i3,i4,i5,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,:,i2,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (3)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,4), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6), mask(i1,i2,:,i3,i4,i5,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,:,i3,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (4)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,5), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6), mask(i1,i2,i3,:,i4,i5,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,:,i4,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (5)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,6), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6), mask(i1,i2,i3,i4,:,i5,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,:,i5,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (6)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,7)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6), mask(i1,i2,i3,i4,i5,:,i6))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,:,i6))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case (7)
        allocate( product_complex_double_st_dim_7D(size(array,1), size(array,2), size(array,3), &
                                               size(array,4), size(array,5), size(array,6)) ) 
        do i6=1,size(product_complex_double_st_dim_7D,6)
          do i5=1,size(product_complex_double_st_dim_7D,5)
            do i4=1,size(product_complex_double_st_dim_7D,4)
              do i3=1,size(product_complex_double_st_dim_7D,3)
                do i2=1,size(product_complex_double_st_dim_7D,2)
                  do i1=1,size(product_complex_double_st_dim_7D,1)
                    if (present(mask)) then
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:), mask(i1,i2,i3,i4,i5,i6,:))
                    else
                      product_complex_double_st_dim_7D(i1,i2,i3,i4,i5,i6) = &
                          product(array(i1,i2,i3,i4,i5,i6,:))
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      case default
        allocate( product_complex_double_st_dim_7D(0,0,0,0,0,0) ) 
        write(ERROR_UNIT, *) "Incorrect value of DIM argument :"
        write(ERROR_UNIT, *) "a zero-sized array has been returned."
    end select
  end function product_complex_double_st_dim_7D




end module cadna_cmplx_array

