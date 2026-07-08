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
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128xui32>
    %32 = stablehlo.convert %31 : (tensor<128xui32>) -> tensor<128xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<i1>
    %37 = stablehlo.constant dense<0.0> : tensor<f32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.not %39 : tensor<i1>
      %45 = stablehlo.and %44, %43 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.dynamic_slice %27, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
      %48 = stablehlo.dynamic_slice %34, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %49 = stablehlo.reshape %48 : (tensor<1xf32>) -> tensor<f32>
      %50 = stablehlo.multiply %13, %47 : tensor<f32>
      %51 = stablehlo.add %4, %50 : tensor<f32>
      %52 = stablehlo.multiply %51, %51 : tensor<f32>
      %53 = stablehlo.multiply %52, %51 : tensor<f32>
      %54 = stablehlo.multiply %9, %53 : tensor<f32>
      %55 = stablehlo.constant dense<0.5> : tensor<f32>
      %56 = stablehlo.multiply %47, %47 : tensor<f32>
      %57 = stablehlo.multiply %55, %56 : tensor<f32>
      %58 = stablehlo.multiply %9, %53 : tensor<f32>
      %59 = stablehlo.negate %58 : tensor<f32>
      %60 = stablehlo.log %53 : tensor<f32>
      %61 = stablehlo.multiply %9, %60 : tensor<f32>
      %62 = stablehlo.add %57, %9 : tensor<f32>
      %63 = stablehlo.add %62, %59 : tensor<f32>
      %64 = stablehlo.add %63, %61 : tensor<f32>
      %65 = stablehlo.log %49 : tensor<f32>
      %66 = stablehlo.compare LT, %65, %64 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %67 = stablehlo.compare GT, %53, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %68 = stablehlo.and %66, %67 : tensor<i1>
      %69 = stablehlo.constant dense<1> : tensor<i32>
      %70 = stablehlo.add %38, %69 : tensor<i32>
      stablehlo.return %70, %68, %54 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %71, %72 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %73 = stablehlo.constant dense<9> : tensor<ui32>
    %74 = stablehlo.shift_right_logical %72, %73 : tensor<ui32>
    %75 = stablehlo.convert %74 : (tensor<ui32>) -> tensor<f32>
    %76 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %77 = stablehlo.multiply %75, %76 : tensor<f32>
    %78 = stablehlo.divide %4, %2 : tensor<f32>
    %79 = stablehlo.power %77, %78 : tensor<f32>
    %80 = stablehlo.select %5, %79, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %81 = stablehlo.multiply %41#2, %80 : tensor<f32>
    %82 = stablehlo.divide %81, %0 : tensor<f32>
    %83 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %84 = stablehlo.reshape %83 : (tensor<1xf32>) -> tensor<f32>
    %85 = stablehlo.constant dense<0.0> : tensor<f32>
    %86 = stablehlo.constant dense<1.0> : tensor<f32>
    %87 = stablehlo.compare LT, %84, %86 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %88 = stablehlo.add %84, %86 : tensor<f32>
    %89 = stablehlo.select %87, %88, %84 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %90 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %91 = stablehlo.subtract %89, %90 : tensor<f32>
    %92 = stablehlo.constant dense<9.0> : tensor<f32>
    %93 = stablehlo.multiply %92, %91 : tensor<f32>
    %94 = stablehlo.sqrt %93 : tensor<f32>
    %95 = stablehlo.divide %86, %94 : tensor<f32>
    %96, %97 = stablehlo.rng_bit_generator %71, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %98 = stablehlo.constant dense<9> : tensor<128xui32>
    %99 = stablehlo.shift_right_logical %97, %98 : tensor<128xui32>
    %100 = stablehlo.convert %99 : (tensor<128xui32>) -> tensor<128xf32>
    %101 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %102 = stablehlo.multiply %100, %101 : tensor<128xf32>
    %103 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %104 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %105 = stablehlo.multiply %102, %103 : tensor<128xf32>
    %106 = stablehlo.subtract %105, %104 : tensor<128xf32>
    %107 = chlo.erf_inv %106 : tensor<128xf32> -> tensor<128xf32>
    %108 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %109 = stablehlo.multiply %107, %108 : tensor<128xf32>
    %110, %111 = stablehlo.rng_bit_generator %96, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %112 = stablehlo.constant dense<9> : tensor<128xui32>
    %113 = stablehlo.shift_right_logical %111, %112 : tensor<128xui32>
    %114 = stablehlo.convert %113 : (tensor<128xui32>) -> tensor<128xf32>
    %115 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %116 = stablehlo.multiply %114, %115 : tensor<128xf32>
    %117 = stablehlo.constant dense<0> : tensor<i32>
    %118 = stablehlo.constant dense<false> : tensor<i1>
    %119 = stablehlo.constant dense<0.0> : tensor<f32>
    %123:3 = stablehlo.while(%120 = %117, %121 = %118, %122 = %119) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %124 = stablehlo.constant dense<128> : tensor<i32>
      %125 = stablehlo.compare LT, %120, %124, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %126 = stablehlo.not %121 : tensor<i1>
      %127 = stablehlo.and %126, %125 : tensor<i1>
      stablehlo.return %127 : tensor<i1>
    } do {
      %128 = stablehlo.dynamic_slice %109, %120, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %129 = stablehlo.reshape %128 : (tensor<1xf32>) -> tensor<f32>
      %130 = stablehlo.dynamic_slice %116, %120, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %131 = stablehlo.reshape %130 : (tensor<1xf32>) -> tensor<f32>
      %132 = stablehlo.multiply %95, %129 : tensor<f32>
      %133 = stablehlo.add %86, %132 : tensor<f32>
      %134 = stablehlo.multiply %133, %133 : tensor<f32>
      %135 = stablehlo.multiply %134, %133 : tensor<f32>
      %136 = stablehlo.multiply %91, %135 : tensor<f32>
      %137 = stablehlo.constant dense<0.5> : tensor<f32>
      %138 = stablehlo.multiply %129, %129 : tensor<f32>
      %139 = stablehlo.multiply %137, %138 : tensor<f32>
      %140 = stablehlo.multiply %91, %135 : tensor<f32>
      %141 = stablehlo.negate %140 : tensor<f32>
      %142 = stablehlo.log %135 : tensor<f32>
      %143 = stablehlo.multiply %91, %142 : tensor<f32>
      %144 = stablehlo.add %139, %91 : tensor<f32>
      %145 = stablehlo.add %144, %141 : tensor<f32>
      %146 = stablehlo.add %145, %143 : tensor<f32>
      %147 = stablehlo.log %131 : tensor<f32>
      %148 = stablehlo.compare LT, %147, %146 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %149 = stablehlo.compare GT, %135, %85 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %150 = stablehlo.and %148, %149 : tensor<i1>
      %151 = stablehlo.constant dense<1> : tensor<i32>
      %152 = stablehlo.add %120, %151 : tensor<i32>
      stablehlo.return %152, %150, %136 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %153, %154 = stablehlo.rng_bit_generator %110, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %155 = stablehlo.constant dense<9> : tensor<ui32>
    %156 = stablehlo.shift_right_logical %154, %155 : tensor<ui32>
    %157 = stablehlo.convert %156 : (tensor<ui32>) -> tensor<f32>
    %158 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %159 = stablehlo.multiply %157, %158 : tensor<f32>
    %160 = stablehlo.divide %86, %84 : tensor<f32>
    %161 = stablehlo.power %159, %160 : tensor<f32>
    %162 = stablehlo.select %87, %161, %86 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %163 = stablehlo.multiply %123#2, %162 : tensor<f32>
    %164 = stablehlo.divide %163, %0 : tensor<f32>
    %165 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %166 = stablehlo.reshape %165 : (tensor<1xf32>) -> tensor<f32>
    %167 = stablehlo.constant dense<0.0> : tensor<f32>
    %168 = stablehlo.constant dense<1.0> : tensor<f32>
    %169 = stablehlo.compare LT, %166, %168 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %170 = stablehlo.add %166, %168 : tensor<f32>
    %171 = stablehlo.select %169, %170, %166 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %172 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %173 = stablehlo.subtract %171, %172 : tensor<f32>
    %174 = stablehlo.constant dense<9.0> : tensor<f32>
    %175 = stablehlo.multiply %174, %173 : tensor<f32>
    %176 = stablehlo.sqrt %175 : tensor<f32>
    %177 = stablehlo.divide %168, %176 : tensor<f32>
    %178, %179 = stablehlo.rng_bit_generator %153, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %180 = stablehlo.constant dense<9> : tensor<128xui32>
    %181 = stablehlo.shift_right_logical %179, %180 : tensor<128xui32>
    %182 = stablehlo.convert %181 : (tensor<128xui32>) -> tensor<128xf32>
    %183 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %184 = stablehlo.multiply %182, %183 : tensor<128xf32>
    %185 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %186 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %187 = stablehlo.multiply %184, %185 : tensor<128xf32>
    %188 = stablehlo.subtract %187, %186 : tensor<128xf32>
    %189 = chlo.erf_inv %188 : tensor<128xf32> -> tensor<128xf32>
    %190 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %191 = stablehlo.multiply %189, %190 : tensor<128xf32>
    %192, %193 = stablehlo.rng_bit_generator %178, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %194 = stablehlo.constant dense<9> : tensor<128xui32>
    %195 = stablehlo.shift_right_logical %193, %194 : tensor<128xui32>
    %196 = stablehlo.convert %195 : (tensor<128xui32>) -> tensor<128xf32>
    %197 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %198 = stablehlo.multiply %196, %197 : tensor<128xf32>
    %199 = stablehlo.constant dense<0> : tensor<i32>
    %200 = stablehlo.constant dense<false> : tensor<i1>
    %201 = stablehlo.constant dense<0.0> : tensor<f32>
    %205:3 = stablehlo.while(%202 = %199, %203 = %200, %204 = %201) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %206 = stablehlo.constant dense<128> : tensor<i32>
      %207 = stablehlo.compare LT, %202, %206, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %208 = stablehlo.not %203 : tensor<i1>
      %209 = stablehlo.and %208, %207 : tensor<i1>
      stablehlo.return %209 : tensor<i1>
    } do {
      %210 = stablehlo.dynamic_slice %191, %202, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %211 = stablehlo.reshape %210 : (tensor<1xf32>) -> tensor<f32>
      %212 = stablehlo.dynamic_slice %198, %202, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %213 = stablehlo.reshape %212 : (tensor<1xf32>) -> tensor<f32>
      %214 = stablehlo.multiply %177, %211 : tensor<f32>
      %215 = stablehlo.add %168, %214 : tensor<f32>
      %216 = stablehlo.multiply %215, %215 : tensor<f32>
      %217 = stablehlo.multiply %216, %215 : tensor<f32>
      %218 = stablehlo.multiply %173, %217 : tensor<f32>
      %219 = stablehlo.constant dense<0.5> : tensor<f32>
      %220 = stablehlo.multiply %211, %211 : tensor<f32>
      %221 = stablehlo.multiply %219, %220 : tensor<f32>
      %222 = stablehlo.multiply %173, %217 : tensor<f32>
      %223 = stablehlo.negate %222 : tensor<f32>
      %224 = stablehlo.log %217 : tensor<f32>
      %225 = stablehlo.multiply %173, %224 : tensor<f32>
      %226 = stablehlo.add %221, %173 : tensor<f32>
      %227 = stablehlo.add %226, %223 : tensor<f32>
      %228 = stablehlo.add %227, %225 : tensor<f32>
      %229 = stablehlo.log %213 : tensor<f32>
      %230 = stablehlo.compare LT, %229, %228 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %231 = stablehlo.compare GT, %217, %167 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %232 = stablehlo.and %230, %231 : tensor<i1>
      %233 = stablehlo.constant dense<1> : tensor<i32>
      %234 = stablehlo.add %202, %233 : tensor<i32>
      stablehlo.return %234, %232, %218 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %235, %236 = stablehlo.rng_bit_generator %192, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %237 = stablehlo.constant dense<9> : tensor<ui32>
    %238 = stablehlo.shift_right_logical %236, %237 : tensor<ui32>
    %239 = stablehlo.convert %238 : (tensor<ui32>) -> tensor<f32>
    %240 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %241 = stablehlo.multiply %239, %240 : tensor<f32>
    %242 = stablehlo.divide %168, %166 : tensor<f32>
    %243 = stablehlo.power %241, %242 : tensor<f32>
    %244 = stablehlo.select %169, %243, %168 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %245 = stablehlo.multiply %205#2, %244 : tensor<f32>
    %246 = stablehlo.divide %245, %0 : tensor<f32>
    %247 = stablehlo.reshape %82 : (tensor<f32>) -> tensor<1xf32>
    %248 = stablehlo.reshape %164 : (tensor<f32>) -> tensor<1xf32>
    %249 = stablehlo.reshape %246 : (tensor<f32>) -> tensor<1xf32>
    %250 = stablehlo.concatenate %247, %248, %249, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %251 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %252 = stablehlo.reduce(%250 init: %251) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %253 = stablehlo.broadcast_in_dim %252, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %254 = stablehlo.divide %250, %253 : tensor<3xf32>
    return %254, %235 : tensor<3xf32>, tensor<2xui64>
  }
}
