module attributes {flatppl.runtime_status = 2 : i64} {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xui64>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %2 = stablehlo.compare LE, %arg0, %1 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %3 = stablehlo.constant dense<false> : tensor<i1>
    %4 = stablehlo.reduce(%2 init: %3) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<9> : tensor<i32>
    %7 = stablehlo.select %4, %6, %5 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.slice %arg0 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.compare LT, %10, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %12 = stablehlo.add %10, %8 : tensor<f32>
    %13 = stablehlo.select %11, %12, %10 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %14 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %15 = stablehlo.subtract %13, %14 : tensor<f32>
    %16 = stablehlo.constant dense<9.0> : tensor<f32>
    %17 = stablehlo.multiply %16, %15 : tensor<f32>
    %18 = stablehlo.sqrt %17 : tensor<f32>
    %19 = stablehlo.divide %8, %18 : tensor<f32>
    %20, %21 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %22 = stablehlo.constant dense<9> : tensor<128xui32>
    %23 = stablehlo.shift_right_logical %21, %22 : tensor<128xui32>
    %24 = stablehlo.convert %23 : (tensor<128xui32>) -> tensor<128xf32>
    %25 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128xf32>
    %27 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %28 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %29 = stablehlo.multiply %26, %27 : tensor<128xf32>
    %30 = stablehlo.subtract %29, %28 : tensor<128xf32>
    %31 = chlo.erf_inv %30 : tensor<128xf32> -> tensor<128xf32>
    %32 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128xf32>
    %34, %35 = stablehlo.rng_bit_generator %20, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %36 = stablehlo.constant dense<9> : tensor<128xui32>
    %37 = stablehlo.shift_right_logical %35, %36 : tensor<128xui32>
    %38 = stablehlo.convert %37 : (tensor<128xui32>) -> tensor<128xf32>
    %39 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %40 = stablehlo.multiply %38, %39 : tensor<128xf32>
    %44:3 = stablehlo.while(%41 = %5, %42 = %3, %43 = %0) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %45 = stablehlo.constant dense<128> : tensor<i32>
      %46 = stablehlo.compare LT, %41, %45, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %47 = stablehlo.not %42 : tensor<i1>
      %48 = stablehlo.and %47, %46 : tensor<i1>
      stablehlo.return %48 : tensor<i1>
    } do {
      %49 = stablehlo.dynamic_slice %33, %41, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %50 = stablehlo.reshape %49 : (tensor<1xf32>) -> tensor<f32>
      %51 = stablehlo.dynamic_slice %40, %41, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %52 = stablehlo.reshape %51 : (tensor<1xf32>) -> tensor<f32>
      %53 = stablehlo.multiply %19, %50 : tensor<f32>
      %54 = stablehlo.add %8, %53 : tensor<f32>
      %55 = stablehlo.multiply %54, %54 : tensor<f32>
      %56 = stablehlo.multiply %55, %54 : tensor<f32>
      %57 = stablehlo.multiply %15, %56 : tensor<f32>
      %58 = stablehlo.constant dense<0.5> : tensor<f32>
      %59 = stablehlo.multiply %50, %50 : tensor<f32>
      %60 = stablehlo.multiply %58, %59 : tensor<f32>
      %61 = stablehlo.negate %57 : tensor<f32>
      %62 = stablehlo.log %56 : tensor<f32>
      %63 = stablehlo.multiply %15, %62 : tensor<f32>
      %64 = stablehlo.add %60, %15 : tensor<f32>
      %65 = stablehlo.add %64, %61 : tensor<f32>
      %66 = stablehlo.add %65, %63 : tensor<f32>
      %67 = stablehlo.log %52 : tensor<f32>
      %68 = stablehlo.compare LT, %67, %66 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %69 = stablehlo.compare GT, %56, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %70 = stablehlo.and %68, %69 : tensor<i1>
      %71 = stablehlo.constant dense<1> : tensor<i32>
      %72 = stablehlo.add %41, %71 : tensor<i32>
      stablehlo.return %72, %70, %57 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %73, %74 = stablehlo.rng_bit_generator %34, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %75 = stablehlo.constant dense<9> : tensor<ui32>
    %76 = stablehlo.shift_right_logical %74, %75 : tensor<ui32>
    %77 = stablehlo.convert %76 : (tensor<ui32>) -> tensor<f32>
    %78 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %79 = stablehlo.multiply %77, %78 : tensor<f32>
    %80 = stablehlo.divide %8, %10 : tensor<f32>
    %81 = stablehlo.power %79, %80 : tensor<f32>
    %82 = stablehlo.select %11, %81, %8 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %83 = stablehlo.multiply %44#2, %82 : tensor<f32>
    %84 = stablehlo.divide %83, %8 : tensor<f32>
    %85 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %86 = stablehlo.reshape %85 : (tensor<1xf32>) -> tensor<f32>
    %87 = stablehlo.compare LT, %86, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %88 = stablehlo.add %86, %8 : tensor<f32>
    %89 = stablehlo.select %87, %88, %86 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %90 = stablehlo.subtract %89, %14 : tensor<f32>
    %91 = stablehlo.multiply %16, %90 : tensor<f32>
    %92 = stablehlo.sqrt %91 : tensor<f32>
    %93 = stablehlo.divide %8, %92 : tensor<f32>
    %94, %95 = stablehlo.rng_bit_generator %73, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %96 = stablehlo.constant dense<9> : tensor<128xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<128xui32>
    %98 = stablehlo.convert %97 : (tensor<128xui32>) -> tensor<128xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<128xf32>
    %101 = stablehlo.multiply %100, %27 : tensor<128xf32>
    %102 = stablehlo.subtract %101, %28 : tensor<128xf32>
    %103 = chlo.erf_inv %102 : tensor<128xf32> -> tensor<128xf32>
    %104 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %105 = stablehlo.multiply %103, %104 : tensor<128xf32>
    %106, %107 = stablehlo.rng_bit_generator %94, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %108 = stablehlo.constant dense<9> : tensor<128xui32>
    %109 = stablehlo.shift_right_logical %107, %108 : tensor<128xui32>
    %110 = stablehlo.convert %109 : (tensor<128xui32>) -> tensor<128xf32>
    %111 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %112 = stablehlo.multiply %110, %111 : tensor<128xf32>
    %116:3 = stablehlo.while(%113 = %5, %114 = %3, %115 = %0) : tensor<i32>, tensor<i1>, tensor<f32>
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
      %126 = stablehlo.add %8, %125 : tensor<f32>
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
      %141 = stablehlo.compare GT, %128, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
    %152 = stablehlo.divide %8, %86 : tensor<f32>
    %153 = stablehlo.power %151, %152 : tensor<f32>
    %154 = stablehlo.select %87, %153, %8 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %155 = stablehlo.multiply %116#2, %154 : tensor<f32>
    %156 = stablehlo.divide %155, %8 : tensor<f32>
    %157 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %158 = stablehlo.reshape %157 : (tensor<1xf32>) -> tensor<f32>
    %159 = stablehlo.compare LT, %158, %8 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %160 = stablehlo.add %158, %8 : tensor<f32>
    %161 = stablehlo.select %159, %160, %158 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %162 = stablehlo.subtract %161, %14 : tensor<f32>
    %163 = stablehlo.multiply %16, %162 : tensor<f32>
    %164 = stablehlo.sqrt %163 : tensor<f32>
    %165 = stablehlo.divide %8, %164 : tensor<f32>
    %166, %167 = stablehlo.rng_bit_generator %145, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %168 = stablehlo.constant dense<9> : tensor<128xui32>
    %169 = stablehlo.shift_right_logical %167, %168 : tensor<128xui32>
    %170 = stablehlo.convert %169 : (tensor<128xui32>) -> tensor<128xf32>
    %171 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %172 = stablehlo.multiply %170, %171 : tensor<128xf32>
    %173 = stablehlo.multiply %172, %27 : tensor<128xf32>
    %174 = stablehlo.subtract %173, %28 : tensor<128xf32>
    %175 = chlo.erf_inv %174 : tensor<128xf32> -> tensor<128xf32>
    %176 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %177 = stablehlo.multiply %175, %176 : tensor<128xf32>
    %178, %179 = stablehlo.rng_bit_generator %166, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %180 = stablehlo.constant dense<9> : tensor<128xui32>
    %181 = stablehlo.shift_right_logical %179, %180 : tensor<128xui32>
    %182 = stablehlo.convert %181 : (tensor<128xui32>) -> tensor<128xf32>
    %183 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %184 = stablehlo.multiply %182, %183 : tensor<128xf32>
    %188:3 = stablehlo.while(%185 = %5, %186 = %3, %187 = %0) : tensor<i32>, tensor<i1>, tensor<f32>
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
      %198 = stablehlo.add %8, %197 : tensor<f32>
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
      %213 = stablehlo.compare GT, %200, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
    %224 = stablehlo.divide %8, %158 : tensor<f32>
    %225 = stablehlo.power %223, %224 : tensor<f32>
    %226 = stablehlo.select %159, %225, %8 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %227 = stablehlo.multiply %188#2, %226 : tensor<f32>
    %228 = stablehlo.divide %227, %8 : tensor<f32>
    %229 = stablehlo.reshape %84 : (tensor<f32>) -> tensor<1xf32>
    %230 = stablehlo.reshape %156 : (tensor<f32>) -> tensor<1xf32>
    %231 = stablehlo.reshape %228 : (tensor<f32>) -> tensor<1xf32>
    %232 = stablehlo.concatenate %229, %230, %231, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %233 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %234 = stablehlo.reduce(%232 init: %233) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %235 = stablehlo.broadcast_in_dim %234, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %236 = stablehlo.divide %232, %235 : tensor<3xf32>
    return %236, %217, %7 : tensor<3xf32>, tensor<2xui64>, tensor<i32>
  }
}
