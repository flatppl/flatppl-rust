module {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %2, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %2, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128xui32>
    %18 = stablehlo.convert %17 : (tensor<128xui32>) -> tensor<128xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128xf32>
    %25 = chlo.erf_inv %24 : tensor<128xf32> -> tensor<128xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128xf32>
    %28 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %29 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128xf32>
    %31 = stablehlo.add %30, %29 : tensor<128xf32>
    %32, %33 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %34 = stablehlo.constant dense<9> : tensor<128xui32>
    %35 = stablehlo.shift_right_logical %33, %34 : tensor<128xui32>
    %36 = stablehlo.convert %35 : (tensor<128xui32>) -> tensor<128xf32>
    %37 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %38 = stablehlo.multiply %36, %37 : tensor<128xf32>
    %39 = stablehlo.subtract %4, %3 : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %42 = stablehlo.multiply %38, %40 : tensor<128xf32>
    %43 = stablehlo.add %42, %41 : tensor<128xf32>
    %44 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.constant dense<false> : tensor<i1>
    %46 = stablehlo.constant dense<0.0> : tensor<f32>
    %50:3 = stablehlo.while(%47 = %44, %48 = %45, %49 = %46) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %51 = stablehlo.constant dense<128> : tensor<i32>
      %52 = stablehlo.compare LT, %47, %51, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %53 = stablehlo.not %48 : tensor<i1>
      %54 = stablehlo.and %53, %52 : tensor<i1>
      stablehlo.return %54 : tensor<i1>
    } do {
      %55 = stablehlo.dynamic_slice %31, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %56 = stablehlo.reshape %55 : (tensor<1xf32>) -> tensor<f32>
      %57 = stablehlo.dynamic_slice %43, %47, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %58 = stablehlo.reshape %57 : (tensor<1xf32>) -> tensor<f32>
      %59 = stablehlo.multiply %13, %56 : tensor<f32>
      %60 = stablehlo.add %4, %59 : tensor<f32>
      %61 = stablehlo.multiply %60, %60 : tensor<f32>
      %62 = stablehlo.multiply %61, %60 : tensor<f32>
      %63 = stablehlo.multiply %9, %62 : tensor<f32>
      %64 = stablehlo.constant dense<0.5> : tensor<f32>
      %65 = stablehlo.multiply %56, %56 : tensor<f32>
      %66 = stablehlo.multiply %64, %65 : tensor<f32>
      %67 = stablehlo.multiply %9, %62 : tensor<f32>
      %68 = stablehlo.negate %67 : tensor<f32>
      %69 = stablehlo.log %62 : tensor<f32>
      %70 = stablehlo.multiply %9, %69 : tensor<f32>
      %71 = stablehlo.add %66, %9 : tensor<f32>
      %72 = stablehlo.add %71, %68 : tensor<f32>
      %73 = stablehlo.add %72, %70 : tensor<f32>
      %74 = stablehlo.log %58 : tensor<f32>
      %75 = stablehlo.compare LT, %74, %73 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %76 = stablehlo.compare GT, %62, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %77 = stablehlo.and %75, %76 : tensor<i1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %47, %78 : tensor<i32>
      stablehlo.return %79, %77, %63 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %80, %81 = stablehlo.rng_bit_generator %32, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %82 = stablehlo.constant dense<9> : tensor<ui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<ui32>
    %84 = stablehlo.convert %83 : (tensor<ui32>) -> tensor<f32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %86 = stablehlo.multiply %84, %85 : tensor<f32>
    %87 = stablehlo.subtract %4, %3 : tensor<f32>
    %88 = stablehlo.multiply %86, %87 : tensor<f32>
    %89 = stablehlo.add %88, %3 : tensor<f32>
    %90 = stablehlo.divide %4, %2 : tensor<f32>
    %91 = stablehlo.power %89, %90 : tensor<f32>
    %92 = stablehlo.select %5, %91, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %93 = stablehlo.multiply %50#2, %92 : tensor<f32>
    %94 = stablehlo.divide %93, %0 : tensor<f32>
    %95 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1xf32>) -> tensor<f32>
    %97 = stablehlo.constant dense<0.0> : tensor<f32>
    %98 = stablehlo.constant dense<1.0> : tensor<f32>
    %99 = stablehlo.compare LT, %96, %98 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %100 = stablehlo.add %96, %98 : tensor<f32>
    %101 = stablehlo.select %99, %100, %96 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %102 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %103 = stablehlo.subtract %101, %102 : tensor<f32>
    %104 = stablehlo.constant dense<9.0> : tensor<f32>
    %105 = stablehlo.multiply %104, %103 : tensor<f32>
    %106 = stablehlo.sqrt %105 : tensor<f32>
    %107 = stablehlo.divide %98, %106 : tensor<f32>
    %108, %109 = stablehlo.rng_bit_generator %80, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %110 = stablehlo.constant dense<9> : tensor<128xui32>
    %111 = stablehlo.shift_right_logical %109, %110 : tensor<128xui32>
    %112 = stablehlo.convert %111 : (tensor<128xui32>) -> tensor<128xf32>
    %113 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<128xf32>
    %115 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %116 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %117 = stablehlo.multiply %114, %115 : tensor<128xf32>
    %118 = stablehlo.subtract %117, %116 : tensor<128xf32>
    %119 = chlo.erf_inv %118 : tensor<128xf32> -> tensor<128xf32>
    %120 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %121 = stablehlo.multiply %119, %120 : tensor<128xf32>
    %122 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %123 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %124 = stablehlo.multiply %121, %122 : tensor<128xf32>
    %125 = stablehlo.add %124, %123 : tensor<128xf32>
    %126, %127 = stablehlo.rng_bit_generator %108, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %128 = stablehlo.constant dense<9> : tensor<128xui32>
    %129 = stablehlo.shift_right_logical %127, %128 : tensor<128xui32>
    %130 = stablehlo.convert %129 : (tensor<128xui32>) -> tensor<128xf32>
    %131 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %132 = stablehlo.multiply %130, %131 : tensor<128xf32>
    %133 = stablehlo.subtract %98, %97 : tensor<f32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %135 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %136 = stablehlo.multiply %132, %134 : tensor<128xf32>
    %137 = stablehlo.add %136, %135 : tensor<128xf32>
    %138 = stablehlo.constant dense<0> : tensor<i32>
    %139 = stablehlo.constant dense<false> : tensor<i1>
    %140 = stablehlo.constant dense<0.0> : tensor<f32>
    %144:3 = stablehlo.while(%141 = %138, %142 = %139, %143 = %140) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %145 = stablehlo.constant dense<128> : tensor<i32>
      %146 = stablehlo.compare LT, %141, %145, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %147 = stablehlo.not %142 : tensor<i1>
      %148 = stablehlo.and %147, %146 : tensor<i1>
      stablehlo.return %148 : tensor<i1>
    } do {
      %149 = stablehlo.dynamic_slice %125, %141, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %150 = stablehlo.reshape %149 : (tensor<1xf32>) -> tensor<f32>
      %151 = stablehlo.dynamic_slice %137, %141, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %152 = stablehlo.reshape %151 : (tensor<1xf32>) -> tensor<f32>
      %153 = stablehlo.multiply %107, %150 : tensor<f32>
      %154 = stablehlo.add %98, %153 : tensor<f32>
      %155 = stablehlo.multiply %154, %154 : tensor<f32>
      %156 = stablehlo.multiply %155, %154 : tensor<f32>
      %157 = stablehlo.multiply %103, %156 : tensor<f32>
      %158 = stablehlo.constant dense<0.5> : tensor<f32>
      %159 = stablehlo.multiply %150, %150 : tensor<f32>
      %160 = stablehlo.multiply %158, %159 : tensor<f32>
      %161 = stablehlo.multiply %103, %156 : tensor<f32>
      %162 = stablehlo.negate %161 : tensor<f32>
      %163 = stablehlo.log %156 : tensor<f32>
      %164 = stablehlo.multiply %103, %163 : tensor<f32>
      %165 = stablehlo.add %160, %103 : tensor<f32>
      %166 = stablehlo.add %165, %162 : tensor<f32>
      %167 = stablehlo.add %166, %164 : tensor<f32>
      %168 = stablehlo.log %152 : tensor<f32>
      %169 = stablehlo.compare LT, %168, %167 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %170 = stablehlo.compare GT, %156, %97 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %171 = stablehlo.and %169, %170 : tensor<i1>
      %172 = stablehlo.constant dense<1> : tensor<i32>
      %173 = stablehlo.add %141, %172 : tensor<i32>
      stablehlo.return %173, %171, %157 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %174, %175 = stablehlo.rng_bit_generator %126, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %176 = stablehlo.constant dense<9> : tensor<ui32>
    %177 = stablehlo.shift_right_logical %175, %176 : tensor<ui32>
    %178 = stablehlo.convert %177 : (tensor<ui32>) -> tensor<f32>
    %179 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %180 = stablehlo.multiply %178, %179 : tensor<f32>
    %181 = stablehlo.subtract %98, %97 : tensor<f32>
    %182 = stablehlo.multiply %180, %181 : tensor<f32>
    %183 = stablehlo.add %182, %97 : tensor<f32>
    %184 = stablehlo.divide %98, %96 : tensor<f32>
    %185 = stablehlo.power %183, %184 : tensor<f32>
    %186 = stablehlo.select %99, %185, %98 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %187 = stablehlo.multiply %144#2, %186 : tensor<f32>
    %188 = stablehlo.divide %187, %0 : tensor<f32>
    %189 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %190 = stablehlo.reshape %189 : (tensor<1xf32>) -> tensor<f32>
    %191 = stablehlo.constant dense<0.0> : tensor<f32>
    %192 = stablehlo.constant dense<1.0> : tensor<f32>
    %193 = stablehlo.compare LT, %190, %192 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %194 = stablehlo.add %190, %192 : tensor<f32>
    %195 = stablehlo.select %193, %194, %190 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %196 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %197 = stablehlo.subtract %195, %196 : tensor<f32>
    %198 = stablehlo.constant dense<9.0> : tensor<f32>
    %199 = stablehlo.multiply %198, %197 : tensor<f32>
    %200 = stablehlo.sqrt %199 : tensor<f32>
    %201 = stablehlo.divide %192, %200 : tensor<f32>
    %202, %203 = stablehlo.rng_bit_generator %174, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %204 = stablehlo.constant dense<9> : tensor<128xui32>
    %205 = stablehlo.shift_right_logical %203, %204 : tensor<128xui32>
    %206 = stablehlo.convert %205 : (tensor<128xui32>) -> tensor<128xf32>
    %207 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %208 = stablehlo.multiply %206, %207 : tensor<128xf32>
    %209 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %210 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %211 = stablehlo.multiply %208, %209 : tensor<128xf32>
    %212 = stablehlo.subtract %211, %210 : tensor<128xf32>
    %213 = chlo.erf_inv %212 : tensor<128xf32> -> tensor<128xf32>
    %214 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %215 = stablehlo.multiply %213, %214 : tensor<128xf32>
    %216 = stablehlo.broadcast_in_dim %192, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %217 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %218 = stablehlo.multiply %215, %216 : tensor<128xf32>
    %219 = stablehlo.add %218, %217 : tensor<128xf32>
    %220, %221 = stablehlo.rng_bit_generator %202, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %222 = stablehlo.constant dense<9> : tensor<128xui32>
    %223 = stablehlo.shift_right_logical %221, %222 : tensor<128xui32>
    %224 = stablehlo.convert %223 : (tensor<128xui32>) -> tensor<128xf32>
    %225 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %226 = stablehlo.multiply %224, %225 : tensor<128xf32>
    %227 = stablehlo.subtract %192, %191 : tensor<f32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %229 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %230 = stablehlo.multiply %226, %228 : tensor<128xf32>
    %231 = stablehlo.add %230, %229 : tensor<128xf32>
    %232 = stablehlo.constant dense<0> : tensor<i32>
    %233 = stablehlo.constant dense<false> : tensor<i1>
    %234 = stablehlo.constant dense<0.0> : tensor<f32>
    %238:3 = stablehlo.while(%235 = %232, %236 = %233, %237 = %234) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %239 = stablehlo.constant dense<128> : tensor<i32>
      %240 = stablehlo.compare LT, %235, %239, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %241 = stablehlo.not %236 : tensor<i1>
      %242 = stablehlo.and %241, %240 : tensor<i1>
      stablehlo.return %242 : tensor<i1>
    } do {
      %243 = stablehlo.dynamic_slice %219, %235, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %244 = stablehlo.reshape %243 : (tensor<1xf32>) -> tensor<f32>
      %245 = stablehlo.dynamic_slice %231, %235, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %246 = stablehlo.reshape %245 : (tensor<1xf32>) -> tensor<f32>
      %247 = stablehlo.multiply %201, %244 : tensor<f32>
      %248 = stablehlo.add %192, %247 : tensor<f32>
      %249 = stablehlo.multiply %248, %248 : tensor<f32>
      %250 = stablehlo.multiply %249, %248 : tensor<f32>
      %251 = stablehlo.multiply %197, %250 : tensor<f32>
      %252 = stablehlo.constant dense<0.5> : tensor<f32>
      %253 = stablehlo.multiply %244, %244 : tensor<f32>
      %254 = stablehlo.multiply %252, %253 : tensor<f32>
      %255 = stablehlo.multiply %197, %250 : tensor<f32>
      %256 = stablehlo.negate %255 : tensor<f32>
      %257 = stablehlo.log %250 : tensor<f32>
      %258 = stablehlo.multiply %197, %257 : tensor<f32>
      %259 = stablehlo.add %254, %197 : tensor<f32>
      %260 = stablehlo.add %259, %256 : tensor<f32>
      %261 = stablehlo.add %260, %258 : tensor<f32>
      %262 = stablehlo.log %246 : tensor<f32>
      %263 = stablehlo.compare LT, %262, %261 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %264 = stablehlo.compare GT, %250, %191 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %265 = stablehlo.and %263, %264 : tensor<i1>
      %266 = stablehlo.constant dense<1> : tensor<i32>
      %267 = stablehlo.add %235, %266 : tensor<i32>
      stablehlo.return %267, %265, %251 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %268, %269 = stablehlo.rng_bit_generator %220, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %270 = stablehlo.constant dense<9> : tensor<ui32>
    %271 = stablehlo.shift_right_logical %269, %270 : tensor<ui32>
    %272 = stablehlo.convert %271 : (tensor<ui32>) -> tensor<f32>
    %273 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %274 = stablehlo.multiply %272, %273 : tensor<f32>
    %275 = stablehlo.subtract %192, %191 : tensor<f32>
    %276 = stablehlo.multiply %274, %275 : tensor<f32>
    %277 = stablehlo.add %276, %191 : tensor<f32>
    %278 = stablehlo.divide %192, %190 : tensor<f32>
    %279 = stablehlo.power %277, %278 : tensor<f32>
    %280 = stablehlo.select %193, %279, %192 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %281 = stablehlo.multiply %238#2, %280 : tensor<f32>
    %282 = stablehlo.divide %281, %0 : tensor<f32>
    %283 = stablehlo.reshape %94 : (tensor<f32>) -> tensor<1xf32>
    %284 = stablehlo.reshape %188 : (tensor<f32>) -> tensor<1xf32>
    %285 = stablehlo.reshape %282 : (tensor<f32>) -> tensor<1xf32>
    %286 = stablehlo.concatenate %283, %284, %285, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %287 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %288 = stablehlo.reduce(%286 init: %287) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %289 = stablehlo.broadcast_in_dim %288, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %290 = stablehlo.divide %286, %289 : tensor<3xf32>
    return %290, %268 : tensor<3xf32>, tensor<2xui64>
  }
}
