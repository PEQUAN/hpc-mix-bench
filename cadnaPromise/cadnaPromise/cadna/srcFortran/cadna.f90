!Copyright 2017 R. CARPENTIER, P. CORDE, F. JEZEQUEL, J.-L. LAMOTTE
!
!This file is part of CADNA.
!
!    CADNA is free software: you can redistribute it and/or modify it
!    under the terms of the GNU Lesser General Public License as
!    published by the Free Software Foundation, either version 3 of the
!    License, or (at your option) any later version.
!
!    CADNA is distributed in the hope that it will be useful, but WITHOUT
!    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
!    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
!    Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with CADNA. If not, see
!    <http://www.gnu.org/licenses/>.
!

module cadna
  use ISO_C_BINDING, only : C_INT
  use cadna_types
  use cadna_add
  use cadna_cmplx_add
  use cadna_sub
  use cadna_cmplx_sub
  use cadna_opp
  use cadna_cmplx_opp
  use cadna_plus
  use cadna_cmplx_plus
  use cadna_mult
  use cadna_cmplx_mult
  use cadna_pow
  use cadna_cmplx_pow
  use cadna_div
  use cadna_cmplx_div
  use cadna_affect
  use cadna_cmplx_affect
  use cadna_lt
  use cadna_le
  use cadna_gt
  use cadna_ge
  use cadna_eq
  use cadna_cmplx_eq
  use cadna_ne
  use cadna_cmplx_ne
  use cadna_math
  use cadna_cmplx_math
  use cadna_array
  use cadna_cmplx_array
  use cadna_convert
  use cadna_cmplx_convert
  use cadna_utils
  use cadna_prec

  interface cadna_init
    subroutine cadna_init_int( max_nb_instability ) bind(C, name="cpp_cadna_init_int")
      import C_INT

      integer(C_INT), VALUE :: max_nb_instability
    end subroutine cadna_init_int
    subroutine cadna_init_int_int( max_nb_instability, &
                                   tag ) bind(C, name="cpp_cadna_init_int_int")
      import C_INT

      integer(C_INT), VALUE :: max_nb_instability
      integer(C_INT), VALUE :: tag
    end subroutine cadna_init_int_int
    subroutine cadna_init_int_int_int_int( max_nb_instability, &
                                           tag,                &
                                           seed,               &
                                           cancellation ) bind(C, name="cpp_cadna_init_int_int_int_int")
      import C_INT

      integer(C_INT), VALUE :: max_nb_instability, tag, seed, cancellation
    end subroutine cadna_init_int_int_int_int
  end interface cadna_init

  interface
    subroutine cadna_end() bind(C, name="cpp_cadna_end")
    end subroutine cadna_end
  end interface

  interface
    subroutine cadna_enable(tag) bind(C, name="cpp_cadna_enable")
      import C_INT
      integer(C_INT), VALUE :: tag
    end subroutine cadna_enable
    subroutine cadna_disable(tag) bind(C, name="cpp_cadna_disable")
      import C_INT
      integer(C_INT), VALUE :: tag
    end subroutine cadna_disable
  end interface

end module cadna
