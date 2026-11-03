module attributes {flatppl.runtime_status = 2 : i64} {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xui64>, tensor<i32>) {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.compare LE, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.slice %arg0 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.compare LT, %9, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %11 = stablehlo.add %9, %7 : tensor<f32>
    %12 = stablehlo.select %10, %11, %9 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %14 = stablehlo.subtract %12, %13 : tensor<f32>
    %15 = stablehlo.constant dense<9.0> : tensor<f32>
    %16 = stablehlo.multiply %15, %14 : tensor<f32>
    %17 = stablehlo.sqrt %16 : tensor<f32>
    %18 = stablehlo.divide %7, %17 : tensor<f32>
    %19, %20 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %21 = stablehlo.constant dense<9> : tensor<128xui32>
    %22 = stablehlo.shift_right_logical %20, %21 : tensor<128xui32>
    %23 = stablehlo.convert %22 : (tensor<128xui32>) -> tensor<128xf32>
    %24 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128xf32>
    %26 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %27 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %28 = stablehlo.multiply %25, %26 : tensor<128xf32>
    %29 = stablehlo.subtract %28, %27 : tensor<128xf32>
    %30 = chlo.erf_inv %29 : tensor<128xf32> -> tensor<128xf32>
    %31 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128xf32>
    %33, %34 = stablehlo.rng_bit_generator %19, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %35 = stablehlo.constant dense<9> : tensor<128xui32>
    %36 = stablehlo.shift_right_logical %34, %35 : tensor<128xui32>
    %37 = stablehlo.convert %36 : (tensor<128xui32>) -> tensor<128xf32>
    %38 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %39 = stablehlo.multiply %37, %38 : tensor<128xf32>
    %40 = stablehlo.constant dense<false> : tensor<i1>
    %44:3 = stablehlo.while(%41 = %4, %42 = %40, %43 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %45 = stablehlo.constant dense<128> : tensor<i32>
      %46 = stablehlo.compare LT, %41, %45, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %47 = stablehlo.not %42 : tensor<i1>
      %48 = stablehlo.and %47, %46 : tensor<i1>
      stablehlo.return %48 : tensor<i1>
    } do {
      %49 = stablehlo.dynamic_slice %32, %41, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %50 = stablehlo.reshape %49 : (tensor<1xf32>) -> tensor<f32>
      %51 = stablehlo.dynamic_slice %39, %41, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %52 = stablehlo.reshape %51 : (tensor<1xf32>) -> tensor<f32>
      %53 = stablehlo.multiply %18, %50 : tensor<f32>
      %54 = stablehlo.add %7, %53 : tensor<f32>
      %55 = stablehlo.multiply %54, %54 : tensor<f32>
      %56 = stablehlo.multiply %55, %54 : tensor<f32>
      %57 = stablehlo.multiply %14, %56 : tensor<f32>
      %58 = stablehlo.constant dense<0.5> : tensor<f32>
      %59 = stablehlo.multiply %50, %50 : tensor<f32>
      %60 = stablehlo.multiply %58, %59 : tensor<f32>
      %61 = stablehlo.negate %57 : tensor<f32>
      %62 = stablehlo.log %56 : tensor<f32>
      %63 = stablehlo.multiply %14, %62 : tensor<f32>
      %64 = stablehlo.add %60, %14 : tensor<f32>
      %65 = stablehlo.add %64, %61 : tensor<f32>
      %66 = stablehlo.add %65, %63 : tensor<f32>
      %67 = stablehlo.log %52 : tensor<f32>
      %68 = stablehlo.compare LT, %67, %66 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %69 = stablehlo.compare GT, %56, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %70 = stablehlo.and %68, %69 : tensor<i1>
      %71 = stablehlo.constant dense<1> : tensor<i32>
      %72 = stablehlo.add %41, %71 : tensor<i32>
      stablehlo.return %72, %70, %57 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %73, %74 = stablehlo.rng_bit_generator %33, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %75 = stablehlo.constant dense<9> : tensor<ui32>
    %76 = stablehlo.shift_right_logical %74, %75 : tensor<ui32>
    %77 = stablehlo.convert %76 : (tensor<ui32>) -> tensor<f32>
    %78 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %79 = stablehlo.multiply %77, %78 : tensor<f32>
    %80 = stablehlo.divide %7, %9 : tensor<f32>
    %81 = stablehlo.power %79, %80 : tensor<f32>
    %82 = stablehlo.select %10, %81, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %83 = stablehlo.multiply %44#2, %82 : tensor<f32>
    %84 = stablehlo.divide %83, %7 : tensor<f32>
    %85 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %86 = stablehlo.reshape %85 : (tensor<1xf32>) -> tensor<f32>
    %87 = stablehlo.compare LT, %86, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %88 = stablehlo.add %86, %7 : tensor<f32>
    %89 = stablehlo.select %87, %88, %86 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %90 = stablehlo.subtract %89, %13 : tensor<f32>
    %91 = stablehlo.multiply %15, %90 : tensor<f32>
    %92 = stablehlo.sqrt %91 : tensor<f32>
    %93 = stablehlo.divide %7, %92 : tensor<f32>
    %94, %95 = stablehlo.rng_bit_generator %73, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %96 = stablehlo.constant dense<9> : tensor<128xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<128xui32>
    %98 = stablehlo.convert %97 : (tensor<128xui32>) -> tensor<128xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<128xf32>
    %101 = stablehlo.multiply %100, %26 : tensor<128xf32>
    %102 = stablehlo.subtract %101, %27 : tensor<128xf32>
    %103 = chlo.erf_inv %102 : tensor<128xf32> -> tensor<128xf32>
    %104 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<128xf32>
    %106, %107 = stablehlo.rng_bit_generator %94, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %108 = stablehlo.constant dense<9> : tensor<128xui32>
    %109 = stablehlo.shift_right_logical %107, %108 : tensor<128xui32>
    %110 = stablehlo.convert %109 : (tensor<128xui32>) -> tensor<128xf32>
    %111 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<128xf32>
    %116:3 = stablehlo.while(%113 = %4, %114 = %40, %115 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %117 = stablehlo.constant dense<128> : tensor<i32>
      %118 = stablehlo.compare LT, %113, %117, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %119 = stablehlo.not %114 : tensor<i1>
      %120 = stablehlo.and %119, %118 : tensor<i1>
      stablehlo.return %120 : tensor<i1>
    } do {
      %121 = stablehlo.dynamic_slice %105, %113, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %122 = stablehlo.reshape %121 : (tensor<1xf32>) -> tensor<f32>
      %123 = stablehlo.dynamic_slice %112, %113, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %124 = stablehlo.reshape %123 : (tensor<1xf32>) -> tensor<f32>
      %125 = stablehlo.multiply %93, %122 : tensor<f32>
      %126 = stablehlo.add %7, %125 : tensor<f32>
      %127 = stablehlo.multiply %126, %126 : tensor<f32>
      %128 = stablehlo.multiply %127, %126 : tensor<f32>
      %129 = stablehlo.multiply %90, %128 : tensor<f32>
      %130 = stablehlo.constant dense<0.5> : tensor<f32>
      %131 = stablehlo.multiply %122, %122 : tensor<f32>
      %132 = stablehlo.multiply %130, %131 : tensor<f32>
      %133 = stablehlo.negate %129 : tensor<f32>
      %134 = stablehlo.log %128 : tensor<f32>
      %135 = stablehlo.multiply %90, %134 : tensor<f32>
      %136 = stablehlo.add %132, %90 : tensor<f32>
      %137 = stablehlo.add %136, %133 : tensor<f32>
      %138 = stablehlo.add %137, %135 : tensor<f32>
      %139 = stablehlo.log %124 : tensor<f32>
      %140 = stablehlo.compare LT, %139, %138 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %141 = stablehlo.compare GT, %128, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %142 = stablehlo.and %140, %141 : tensor<i1>
      %143 = stablehlo.constant dense<1> : tensor<i32>
      %144 = stablehlo.add %113, %143 : tensor<i32>
      stablehlo.return %144, %142, %129 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %145, %146 = stablehlo.rng_bit_generator %106, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %147 = stablehlo.constant dense<9> : tensor<ui32>
    %148 = stablehlo.shift_right_logical %146, %147 : tensor<ui32>
    %149 = stablehlo.convert %148 : (tensor<ui32>) -> tensor<f32>
    %150 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %151 = stablehlo.multiply %149, %150 : tensor<f32>
    %152 = stablehlo.divide %7, %86 : tensor<f32>
    %153 = stablehlo.power %151, %152 : tensor<f32>
    %154 = stablehlo.select %87, %153, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %155 = stablehlo.multiply %116#2, %154 : tensor<f32>
    %156 = stablehlo.divide %155, %7 : tensor<f32>
    %157 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %158 = stablehlo.reshape %157 : (tensor<1xf32>) -> tensor<f32>
    %159 = stablehlo.compare LT, %158, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %160 = stablehlo.add %158, %7 : tensor<f32>
    %161 = stablehlo.select %159, %160, %158 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %162 = stablehlo.subtract %161, %13 : tensor<f32>
    %163 = stablehlo.multiply %15, %162 : tensor<f32>
    %164 = stablehlo.sqrt %163 : tensor<f32>
    %165 = stablehlo.divide %7, %164 : tensor<f32>
    %166, %167 = stablehlo.rng_bit_generator %145, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %168 = stablehlo.constant dense<9> : tensor<128xui32>
    %169 = stablehlo.shift_right_logical %167, %168 : tensor<128xui32>
    %170 = stablehlo.convert %169 : (tensor<128xui32>) -> tensor<128xf32>
    %171 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %172 = stablehlo.multiply %170, %171 : tensor<128xf32>
    %173 = stablehlo.multiply %172, %26 : tensor<128xf32>
    %174 = stablehlo.subtract %173, %27 : tensor<128xf32>
    %175 = chlo.erf_inv %174 : tensor<128xf32> -> tensor<128xf32>
    %176 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %177 = stablehlo.multiply %175, %176 : tensor<128xf32>
    %178, %179 = stablehlo.rng_bit_generator %166, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %180 = stablehlo.constant dense<9> : tensor<128xui32>
    %181 = stablehlo.shift_right_logical %179, %180 : tensor<128xui32>
    %182 = stablehlo.convert %181 : (tensor<128xui32>) -> tensor<128xf32>
    %183 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %184 = stablehlo.multiply %182, %183 : tensor<128xf32>
    %188:3 = stablehlo.while(%185 = %4, %186 = %40, %187 = %2) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %189 = stablehlo.constant dense<128> : tensor<i32>
      %190 = stablehlo.compare LT, %185, %189, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %191 = stablehlo.not %186 : tensor<i1>
      %192 = stablehlo.and %191, %190 : tensor<i1>
      stablehlo.return %192 : tensor<i1>
    } do {
      %193 = stablehlo.dynamic_slice %177, %185, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %194 = stablehlo.reshape %193 : (tensor<1xf32>) -> tensor<f32>
      %195 = stablehlo.dynamic_slice %184, %185, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %196 = stablehlo.reshape %195 : (tensor<1xf32>) -> tensor<f32>
      %197 = stablehlo.multiply %165, %194 : tensor<f32>
      %198 = stablehlo.add %7, %197 : tensor<f32>
      %199 = stablehlo.multiply %198, %198 : tensor<f32>
      %200 = stablehlo.multiply %199, %198 : tensor<f32>
      %201 = stablehlo.multiply %162, %200 : tensor<f32>
      %202 = stablehlo.constant dense<0.5> : tensor<f32>
      %203 = stablehlo.multiply %194, %194 : tensor<f32>
      %204 = stablehlo.multiply %202, %203 : tensor<f32>
      %205 = stablehlo.negate %201 : tensor<f32>
      %206 = stablehlo.log %200 : tensor<f32>
      %207 = stablehlo.multiply %162, %206 : tensor<f32>
      %208 = stablehlo.add %204, %162 : tensor<f32>
      %209 = stablehlo.add %208, %205 : tensor<f32>
      %210 = stablehlo.add %209, %207 : tensor<f32>
      %211 = stablehlo.log %196 : tensor<f32>
      %212 = stablehlo.compare LT, %211, %210 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %213 = stablehlo.compare GT, %200, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %214 = stablehlo.and %212, %213 : tensor<i1>
      %215 = stablehlo.constant dense<1> : tensor<i32>
      %216 = stablehlo.add %185, %215 : tensor<i32>
      stablehlo.return %216, %214, %201 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %217, %218 = stablehlo.rng_bit_generator %178, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %219 = stablehlo.constant dense<9> : tensor<ui32>
    %220 = stablehlo.shift_right_logical %218, %219 : tensor<ui32>
    %221 = stablehlo.convert %220 : (tensor<ui32>) -> tensor<f32>
    %222 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %223 = stablehlo.multiply %221, %222 : tensor<f32>
    %224 = stablehlo.divide %7, %158 : tensor<f32>
    %225 = stablehlo.power %223, %224 : tensor<f32>
    %226 = stablehlo.select %159, %225, %7 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %227 = stablehlo.multiply %188#2, %226 : tensor<f32>
    %228 = stablehlo.divide %227, %7 : tensor<f32>
    %229 = stablehlo.reshape %84 : (tensor<f32>) -> tensor<1xf32>
    %230 = stablehlo.reshape %156 : (tensor<f32>) -> tensor<1xf32>
    %231 = stablehlo.reshape %228 : (tensor<f32>) -> tensor<1xf32>
    %232 = stablehlo.concatenate %229, %230, %231, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %234 = stablehlo.reduce(%232 init: %233) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %236 = stablehlo.divide %232, %235 : tensor<3xf32>
    return %236, %217, %6 : tensor<3xf32>, tensor<2xui64>, tensor<i32>
  }
}
