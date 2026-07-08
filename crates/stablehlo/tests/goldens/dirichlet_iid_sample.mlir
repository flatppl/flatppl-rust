module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<5x3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<4.0> : tensor<f32>
    %3 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %5 = stablehlo.reshape %2 : (tensor<f32>) -> tensor<1xf32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %7 = stablehlo.constant dense<1.0> : tensor<f32>
    %8 = stablehlo.slice %6 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.constant dense<0.0> : tensor<f32>
    %11 = stablehlo.constant dense<1.0> : tensor<f32>
    %12 = stablehlo.compare LT, %9, %11 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %13 = stablehlo.add %9, %11 : tensor<f32>
    %14 = stablehlo.select %12, %13, %9 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %16 = stablehlo.subtract %14, %15 : tensor<f32>
    %17 = stablehlo.constant dense<9.0> : tensor<f32>
    %18 = stablehlo.multiply %17, %16 : tensor<f32>
    %19 = stablehlo.sqrt %18 : tensor<f32>
    %20 = stablehlo.divide %11, %19 : tensor<f32>
    %21, %22 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %23 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<128x5xui32>
    %25 = stablehlo.convert %24 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %26 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x5xf32>
    %28 = stablehlo.constant dense<2.0> : tensor<128x5xf32>
    %29 = stablehlo.constant dense<1.0> : tensor<128x5xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128x5xf32>
    %31 = stablehlo.subtract %30, %29 : tensor<128x5xf32>
    %32 = chlo.erf_inv %31 : tensor<128x5xf32> -> tensor<128x5xf32>
    %33 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128x5xf32>
    %35, %36 = stablehlo.rng_bit_generator %21, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %37 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %38 = stablehlo.shift_right_logical %36, %37 : tensor<128x5xui32>
    %39 = stablehlo.convert %38 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %40 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %41 = stablehlo.multiply %39, %40 : tensor<128x5xf32>
    %42 = stablehlo.constant dense<0> : tensor<i32>
    %43 = stablehlo.constant dense<false> : tensor<5xi1>
    %44 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %48:3 = stablehlo.while(%45 = %42, %46 = %43, %47 = %44) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %49 = stablehlo.constant dense<128> : tensor<i32>
      %50 = stablehlo.compare LT, %45, %49, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %51 = stablehlo.constant dense<true> : tensor<i1>
      %52 = stablehlo.reduce(%46 init: %51) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %53 = stablehlo.not %52 : tensor<i1>
      %54 = stablehlo.and %50, %53 : tensor<i1>
      stablehlo.return %54 : tensor<i1>
    } do {
      %55 = stablehlo.constant dense<0> : tensor<i32>
      %56 = stablehlo.dynamic_slice %34, %45, %55, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %57 = stablehlo.reshape %56 : (tensor<1x5xf32>) -> tensor<5xf32>
      %58 = stablehlo.constant dense<0> : tensor<i32>
      %59 = stablehlo.dynamic_slice %41, %45, %58, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %60 = stablehlo.reshape %59 : (tensor<1x5xf32>) -> tensor<5xf32>
      %61 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %62 = stablehlo.multiply %61, %57 : tensor<5xf32>
      %63 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %64 = stablehlo.add %63, %62 : tensor<5xf32>
      %65 = stablehlo.multiply %64, %64 : tensor<5xf32>
      %66 = stablehlo.multiply %65, %64 : tensor<5xf32>
      %67 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %68 = stablehlo.multiply %67, %66 : tensor<5xf32>
      %69 = stablehlo.constant dense<0.5> : tensor<f32>
      %70 = stablehlo.multiply %57, %57 : tensor<5xf32>
      %71 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %72 = stablehlo.multiply %71, %70 : tensor<5xf32>
      %73 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %74 = stablehlo.multiply %73, %66 : tensor<5xf32>
      %75 = stablehlo.negate %74 : tensor<5xf32>
      %76 = stablehlo.log %66 : tensor<5xf32>
      %77 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %78 = stablehlo.multiply %77, %76 : tensor<5xf32>
      %79 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %80 = stablehlo.add %72, %79 : tensor<5xf32>
      %81 = stablehlo.add %80, %75 : tensor<5xf32>
      %82 = stablehlo.add %81, %78 : tensor<5xf32>
      %83 = stablehlo.log %60 : tensor<5xf32>
      %84 = stablehlo.compare LT, %83, %82 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %85 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %86 = stablehlo.compare GT, %66, %85 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %87 = stablehlo.and %84, %86 : tensor<5xi1>
      %88 = stablehlo.select %46, %47, %68 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %89 = stablehlo.or %46, %87 : tensor<5xi1>
      %90 = stablehlo.constant dense<1> : tensor<i32>
      %91 = stablehlo.add %45, %90 : tensor<i32>
      stablehlo.return %91, %89, %88 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %92, %93 = stablehlo.rng_bit_generator %35, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %94 = stablehlo.constant dense<9> : tensor<5xui32>
    %95 = stablehlo.shift_right_logical %93, %94 : tensor<5xui32>
    %96 = stablehlo.convert %95 : (tensor<5xui32>) -> tensor<5xf32>
    %97 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %98 = stablehlo.multiply %96, %97 : tensor<5xf32>
    %99 = stablehlo.divide %11, %9 : tensor<f32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %101 = stablehlo.power %98, %100 : tensor<5xf32>
    %102 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %103 = stablehlo.select %12, %101, %102 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %104 = stablehlo.multiply %48#2, %103 : tensor<5xf32>
    %105 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %106 = stablehlo.divide %104, %105 : tensor<5xf32>
    %107 = stablehlo.slice %6 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %108 = stablehlo.reshape %107 : (tensor<1xf32>) -> tensor<f32>
    %109 = stablehlo.constant dense<0.0> : tensor<f32>
    %110 = stablehlo.constant dense<1.0> : tensor<f32>
    %111 = stablehlo.compare LT, %108, %110 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %112 = stablehlo.add %108, %110 : tensor<f32>
    %113 = stablehlo.select %111, %112, %108 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %114 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %115 = stablehlo.subtract %113, %114 : tensor<f32>
    %116 = stablehlo.constant dense<9.0> : tensor<f32>
    %117 = stablehlo.multiply %116, %115 : tensor<f32>
    %118 = stablehlo.sqrt %117 : tensor<f32>
    %119 = stablehlo.divide %110, %118 : tensor<f32>
    %120, %121 = stablehlo.rng_bit_generator %92, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %122 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %123 = stablehlo.shift_right_logical %121, %122 : tensor<128x5xui32>
    %124 = stablehlo.convert %123 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %125 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %126 = stablehlo.multiply %124, %125 : tensor<128x5xf32>
    %127 = stablehlo.constant dense<2.0> : tensor<128x5xf32>
    %128 = stablehlo.constant dense<1.0> : tensor<128x5xf32>
    %129 = stablehlo.multiply %126, %127 : tensor<128x5xf32>
    %130 = stablehlo.subtract %129, %128 : tensor<128x5xf32>
    %131 = chlo.erf_inv %130 : tensor<128x5xf32> -> tensor<128x5xf32>
    %132 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %133 = stablehlo.multiply %131, %132 : tensor<128x5xf32>
    %134, %135 = stablehlo.rng_bit_generator %120, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %136 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %137 = stablehlo.shift_right_logical %135, %136 : tensor<128x5xui32>
    %138 = stablehlo.convert %137 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %139 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %140 = stablehlo.multiply %138, %139 : tensor<128x5xf32>
    %141 = stablehlo.constant dense<0> : tensor<i32>
    %142 = stablehlo.constant dense<false> : tensor<5xi1>
    %143 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %147:3 = stablehlo.while(%144 = %141, %145 = %142, %146 = %143) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %148 = stablehlo.constant dense<128> : tensor<i32>
      %149 = stablehlo.compare LT, %144, %148, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %150 = stablehlo.constant dense<true> : tensor<i1>
      %151 = stablehlo.reduce(%145 init: %150) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %152 = stablehlo.not %151 : tensor<i1>
      %153 = stablehlo.and %149, %152 : tensor<i1>
      stablehlo.return %153 : tensor<i1>
    } do {
      %154 = stablehlo.constant dense<0> : tensor<i32>
      %155 = stablehlo.dynamic_slice %133, %144, %154, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %156 = stablehlo.reshape %155 : (tensor<1x5xf32>) -> tensor<5xf32>
      %157 = stablehlo.constant dense<0> : tensor<i32>
      %158 = stablehlo.dynamic_slice %140, %144, %157, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %159 = stablehlo.reshape %158 : (tensor<1x5xf32>) -> tensor<5xf32>
      %160 = stablehlo.broadcast_in_dim %119, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %161 = stablehlo.multiply %160, %156 : tensor<5xf32>
      %162 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %163 = stablehlo.add %162, %161 : tensor<5xf32>
      %164 = stablehlo.multiply %163, %163 : tensor<5xf32>
      %165 = stablehlo.multiply %164, %163 : tensor<5xf32>
      %166 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %167 = stablehlo.multiply %166, %165 : tensor<5xf32>
      %168 = stablehlo.constant dense<0.5> : tensor<f32>
      %169 = stablehlo.multiply %156, %156 : tensor<5xf32>
      %170 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %171 = stablehlo.multiply %170, %169 : tensor<5xf32>
      %172 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %173 = stablehlo.multiply %172, %165 : tensor<5xf32>
      %174 = stablehlo.negate %173 : tensor<5xf32>
      %175 = stablehlo.log %165 : tensor<5xf32>
      %176 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %177 = stablehlo.multiply %176, %175 : tensor<5xf32>
      %178 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %179 = stablehlo.add %171, %178 : tensor<5xf32>
      %180 = stablehlo.add %179, %174 : tensor<5xf32>
      %181 = stablehlo.add %180, %177 : tensor<5xf32>
      %182 = stablehlo.log %159 : tensor<5xf32>
      %183 = stablehlo.compare LT, %182, %181 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %184 = stablehlo.broadcast_in_dim %109, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %185 = stablehlo.compare GT, %165, %184 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %186 = stablehlo.and %183, %185 : tensor<5xi1>
      %187 = stablehlo.select %145, %146, %167 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %188 = stablehlo.or %145, %186 : tensor<5xi1>
      %189 = stablehlo.constant dense<1> : tensor<i32>
      %190 = stablehlo.add %144, %189 : tensor<i32>
      stablehlo.return %190, %188, %187 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %191, %192 = stablehlo.rng_bit_generator %134, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %193 = stablehlo.constant dense<9> : tensor<5xui32>
    %194 = stablehlo.shift_right_logical %192, %193 : tensor<5xui32>
    %195 = stablehlo.convert %194 : (tensor<5xui32>) -> tensor<5xf32>
    %196 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %197 = stablehlo.multiply %195, %196 : tensor<5xf32>
    %198 = stablehlo.divide %110, %108 : tensor<f32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %200 = stablehlo.power %197, %199 : tensor<5xf32>
    %201 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %202 = stablehlo.select %111, %200, %201 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %203 = stablehlo.multiply %147#2, %202 : tensor<5xf32>
    %204 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %205 = stablehlo.divide %203, %204 : tensor<5xf32>
    %206 = stablehlo.slice %6 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %207 = stablehlo.reshape %206 : (tensor<1xf32>) -> tensor<f32>
    %208 = stablehlo.constant dense<0.0> : tensor<f32>
    %209 = stablehlo.constant dense<1.0> : tensor<f32>
    %210 = stablehlo.compare LT, %207, %209 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %211 = stablehlo.add %207, %209 : tensor<f32>
    %212 = stablehlo.select %210, %211, %207 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %213 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %214 = stablehlo.subtract %212, %213 : tensor<f32>
    %215 = stablehlo.constant dense<9.0> : tensor<f32>
    %216 = stablehlo.multiply %215, %214 : tensor<f32>
    %217 = stablehlo.sqrt %216 : tensor<f32>
    %218 = stablehlo.divide %209, %217 : tensor<f32>
    %219, %220 = stablehlo.rng_bit_generator %191, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %221 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %222 = stablehlo.shift_right_logical %220, %221 : tensor<128x5xui32>
    %223 = stablehlo.convert %222 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %224 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %225 = stablehlo.multiply %223, %224 : tensor<128x5xf32>
    %226 = stablehlo.constant dense<2.0> : tensor<128x5xf32>
    %227 = stablehlo.constant dense<1.0> : tensor<128x5xf32>
    %228 = stablehlo.multiply %225, %226 : tensor<128x5xf32>
    %229 = stablehlo.subtract %228, %227 : tensor<128x5xf32>
    %230 = chlo.erf_inv %229 : tensor<128x5xf32> -> tensor<128x5xf32>
    %231 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %232 = stablehlo.multiply %230, %231 : tensor<128x5xf32>
    %233, %234 = stablehlo.rng_bit_generator %219, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %235 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %236 = stablehlo.shift_right_logical %234, %235 : tensor<128x5xui32>
    %237 = stablehlo.convert %236 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %238 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %239 = stablehlo.multiply %237, %238 : tensor<128x5xf32>
    %240 = stablehlo.constant dense<0> : tensor<i32>
    %241 = stablehlo.constant dense<false> : tensor<5xi1>
    %242 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %246:3 = stablehlo.while(%243 = %240, %244 = %241, %245 = %242) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %247 = stablehlo.constant dense<128> : tensor<i32>
      %248 = stablehlo.compare LT, %243, %247, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %249 = stablehlo.constant dense<true> : tensor<i1>
      %250 = stablehlo.reduce(%244 init: %249) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %251 = stablehlo.not %250 : tensor<i1>
      %252 = stablehlo.and %248, %251 : tensor<i1>
      stablehlo.return %252 : tensor<i1>
    } do {
      %253 = stablehlo.constant dense<0> : tensor<i32>
      %254 = stablehlo.dynamic_slice %232, %243, %253, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %255 = stablehlo.reshape %254 : (tensor<1x5xf32>) -> tensor<5xf32>
      %256 = stablehlo.constant dense<0> : tensor<i32>
      %257 = stablehlo.dynamic_slice %239, %243, %256, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %258 = stablehlo.reshape %257 : (tensor<1x5xf32>) -> tensor<5xf32>
      %259 = stablehlo.broadcast_in_dim %218, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %260 = stablehlo.multiply %259, %255 : tensor<5xf32>
      %261 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %262 = stablehlo.add %261, %260 : tensor<5xf32>
      %263 = stablehlo.multiply %262, %262 : tensor<5xf32>
      %264 = stablehlo.multiply %263, %262 : tensor<5xf32>
      %265 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %266 = stablehlo.multiply %265, %264 : tensor<5xf32>
      %267 = stablehlo.constant dense<0.5> : tensor<f32>
      %268 = stablehlo.multiply %255, %255 : tensor<5xf32>
      %269 = stablehlo.broadcast_in_dim %267, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %270 = stablehlo.multiply %269, %268 : tensor<5xf32>
      %271 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %272 = stablehlo.multiply %271, %264 : tensor<5xf32>
      %273 = stablehlo.negate %272 : tensor<5xf32>
      %274 = stablehlo.log %264 : tensor<5xf32>
      %275 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %276 = stablehlo.multiply %275, %274 : tensor<5xf32>
      %277 = stablehlo.broadcast_in_dim %214, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %278 = stablehlo.add %270, %277 : tensor<5xf32>
      %279 = stablehlo.add %278, %273 : tensor<5xf32>
      %280 = stablehlo.add %279, %276 : tensor<5xf32>
      %281 = stablehlo.log %258 : tensor<5xf32>
      %282 = stablehlo.compare LT, %281, %280 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %283 = stablehlo.broadcast_in_dim %208, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %284 = stablehlo.compare GT, %264, %283 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %285 = stablehlo.and %282, %284 : tensor<5xi1>
      %286 = stablehlo.select %244, %245, %266 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %287 = stablehlo.or %244, %285 : tensor<5xi1>
      %288 = stablehlo.constant dense<1> : tensor<i32>
      %289 = stablehlo.add %243, %288 : tensor<i32>
      stablehlo.return %289, %287, %286 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %290, %291 = stablehlo.rng_bit_generator %233, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %292 = stablehlo.constant dense<9> : tensor<5xui32>
    %293 = stablehlo.shift_right_logical %291, %292 : tensor<5xui32>
    %294 = stablehlo.convert %293 : (tensor<5xui32>) -> tensor<5xf32>
    %295 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %296 = stablehlo.multiply %294, %295 : tensor<5xf32>
    %297 = stablehlo.divide %209, %207 : tensor<f32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %299 = stablehlo.power %296, %298 : tensor<5xf32>
    %300 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %301 = stablehlo.select %210, %299, %300 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %302 = stablehlo.multiply %246#2, %301 : tensor<5xf32>
    %303 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %304 = stablehlo.divide %302, %303 : tensor<5xf32>
    %305 = stablehlo.reshape %106 : (tensor<5xf32>) -> tensor<1x5xf32>
    %306 = stablehlo.reshape %205 : (tensor<5xf32>) -> tensor<1x5xf32>
    %307 = stablehlo.reshape %304 : (tensor<5xf32>) -> tensor<1x5xf32>
    %308 = stablehlo.concatenate %305, %306, %307, dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
    %309 = stablehlo.transpose %308, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %310 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %311 = stablehlo.reduce(%309 init: %310) applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %313 = stablehlo.divide %309, %312 : tensor<5x3xf32>
    return %313, %290 : tensor<5x3xf32>, tensor<2xui64>
  }
}
