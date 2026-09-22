module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<5x3xf32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<2.0> : tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.constant dense<0.0> : tensor<f32>
    %5 = stablehlo.compare LT, %3, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.constant dense<3.0> : tensor<f32>
    %7 = stablehlo.select %5, %6, %3 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %1, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128x5xui32>
    %18 = stablehlo.convert %17 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x5xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128x5xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128x5xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128x5xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128x5xf32>
    %25 = chlo.erf_inv %24 : tensor<128x5xf32> -> tensor<128x5xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x5xf32>
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128x5xui32>
    %32 = stablehlo.convert %31 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128x5xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<5xi1>
    %37 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.constant dense<true> : tensor<i1>
      %45 = stablehlo.reduce(%39 init: %44) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %46 = stablehlo.not %45 : tensor<i1>
      %47 = stablehlo.and %43, %46 : tensor<i1>
      stablehlo.return %47 : tensor<i1>
    } do {
      %48 = stablehlo.constant dense<0> : tensor<i32>
      %49 = stablehlo.dynamic_slice %27, %38, %48, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x5xf32>) -> tensor<5xf32>
      %51 = stablehlo.dynamic_slice %34, %38, %48, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %52 = stablehlo.reshape %51 : (tensor<1x5xf32>) -> tensor<5xf32>
      %53 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %54 = stablehlo.multiply %53, %50 : tensor<5xf32>
      %55 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %56 = stablehlo.add %55, %54 : tensor<5xf32>
      %57 = stablehlo.multiply %56, %56 : tensor<5xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<5xf32>
      %59 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %60 = stablehlo.multiply %59, %58 : tensor<5xf32>
      %61 = stablehlo.constant dense<0.5> : tensor<f32>
      %62 = stablehlo.multiply %50, %50 : tensor<5xf32>
      %63 = stablehlo.broadcast_in_dim %61, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %64 = stablehlo.multiply %63, %62 : tensor<5xf32>
      %65 = stablehlo.negate %60 : tensor<5xf32>
      %66 = stablehlo.log %58 : tensor<5xf32>
      %67 = stablehlo.multiply %59, %66 : tensor<5xf32>
      %68 = stablehlo.add %64, %59 : tensor<5xf32>
      %69 = stablehlo.add %68, %65 : tensor<5xf32>
      %70 = stablehlo.add %69, %67 : tensor<5xf32>
      %71 = stablehlo.log %52 : tensor<5xf32>
      %72 = stablehlo.compare LT, %71, %70 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %73 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %74 = stablehlo.compare GT, %58, %73 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %75 = stablehlo.and %72, %74 : tensor<5xi1>
      %76 = stablehlo.select %39, %40, %60 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %77 = stablehlo.or %39, %75 : tensor<5xi1>
      %78 = stablehlo.constant dense<1> : tensor<i32>
      %79 = stablehlo.add %38, %78 : tensor<i32>
      stablehlo.return %79, %77, %76 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %80, %81 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %82 = stablehlo.constant dense<9> : tensor<5xui32>
    %83 = stablehlo.shift_right_logical %81, %82 : tensor<5xui32>
    %84 = stablehlo.convert %83 : (tensor<5xui32>) -> tensor<5xf32>
    %85 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %86 = stablehlo.multiply %84, %85 : tensor<5xf32>
    %87 = stablehlo.constant dense<0.5> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %89 = stablehlo.power %86, %88 : tensor<5xf32>
    %90 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %91 = stablehlo.select %5, %89, %90 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %92 = stablehlo.multiply %41#2, %91 : tensor<5xf32>
    %93 = stablehlo.divide %92, %90 : tensor<5xf32>
    %94 = stablehlo.constant dense<3.0> : tensor<1xf32>
    %95 = stablehlo.reshape %94 : (tensor<1xf32>) -> tensor<f32>
    %96 = stablehlo.compare LT, %95, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %97 = stablehlo.constant dense<4.0> : tensor<f32>
    %98 = stablehlo.select %96, %97, %95 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %99 = stablehlo.subtract %98, %8 : tensor<f32>
    %100 = stablehlo.multiply %10, %99 : tensor<f32>
    %101 = stablehlo.sqrt %100 : tensor<f32>
    %102 = stablehlo.divide %1, %101 : tensor<f32>
    %103, %104 = stablehlo.rng_bit_generator %80, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %105 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %106 = stablehlo.shift_right_logical %104, %105 : tensor<128x5xui32>
    %107 = stablehlo.convert %106 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %108 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %109 = stablehlo.multiply %107, %108 : tensor<128x5xf32>
    %110 = stablehlo.multiply %109, %21 : tensor<128x5xf32>
    %111 = stablehlo.subtract %110, %22 : tensor<128x5xf32>
    %112 = chlo.erf_inv %111 : tensor<128x5xf32> -> tensor<128x5xf32>
    %113 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<128x5xf32>
    %115, %116 = stablehlo.rng_bit_generator %103, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %117 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %118 = stablehlo.shift_right_logical %116, %117 : tensor<128x5xui32>
    %119 = stablehlo.convert %118 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %120 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %121 = stablehlo.multiply %119, %120 : tensor<128x5xf32>
    %122 = stablehlo.constant dense<false> : tensor<5xi1>
    %126:3 = stablehlo.while(%123 = %35, %124 = %122, %125 = %37) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %127 = stablehlo.constant dense<128> : tensor<i32>
      %128 = stablehlo.compare LT, %123, %127, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %129 = stablehlo.constant dense<true> : tensor<i1>
      %130 = stablehlo.reduce(%124 init: %129) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %131 = stablehlo.not %130 : tensor<i1>
      %132 = stablehlo.and %128, %131 : tensor<i1>
      stablehlo.return %132 : tensor<i1>
    } do {
      %133 = stablehlo.constant dense<0> : tensor<i32>
      %134 = stablehlo.dynamic_slice %114, %123, %133, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %135 = stablehlo.reshape %134 : (tensor<1x5xf32>) -> tensor<5xf32>
      %136 = stablehlo.dynamic_slice %121, %123, %133, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %137 = stablehlo.reshape %136 : (tensor<1x5xf32>) -> tensor<5xf32>
      %138 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %139 = stablehlo.multiply %138, %135 : tensor<5xf32>
      %140 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %141 = stablehlo.add %140, %139 : tensor<5xf32>
      %142 = stablehlo.multiply %141, %141 : tensor<5xf32>
      %143 = stablehlo.multiply %142, %141 : tensor<5xf32>
      %144 = stablehlo.broadcast_in_dim %99, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %145 = stablehlo.multiply %144, %143 : tensor<5xf32>
      %146 = stablehlo.constant dense<0.5> : tensor<f32>
      %147 = stablehlo.multiply %135, %135 : tensor<5xf32>
      %148 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %149 = stablehlo.multiply %148, %147 : tensor<5xf32>
      %150 = stablehlo.negate %145 : tensor<5xf32>
      %151 = stablehlo.log %143 : tensor<5xf32>
      %152 = stablehlo.multiply %144, %151 : tensor<5xf32>
      %153 = stablehlo.add %149, %144 : tensor<5xf32>
      %154 = stablehlo.add %153, %150 : tensor<5xf32>
      %155 = stablehlo.add %154, %152 : tensor<5xf32>
      %156 = stablehlo.log %137 : tensor<5xf32>
      %157 = stablehlo.compare LT, %156, %155 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %158 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %159 = stablehlo.compare GT, %143, %158 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %160 = stablehlo.and %157, %159 : tensor<5xi1>
      %161 = stablehlo.select %124, %125, %145 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %162 = stablehlo.or %124, %160 : tensor<5xi1>
      %163 = stablehlo.constant dense<1> : tensor<i32>
      %164 = stablehlo.add %123, %163 : tensor<i32>
      stablehlo.return %164, %162, %161 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %165, %166 = stablehlo.rng_bit_generator %115, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %167 = stablehlo.constant dense<9> : tensor<5xui32>
    %168 = stablehlo.shift_right_logical %166, %167 : tensor<5xui32>
    %169 = stablehlo.convert %168 : (tensor<5xui32>) -> tensor<5xf32>
    %170 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %171 = stablehlo.multiply %169, %170 : tensor<5xf32>
    %172 = stablehlo.constant dense<0.3333333432674408> : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %174 = stablehlo.power %171, %173 : tensor<5xf32>
    %175 = stablehlo.select %96, %174, %90 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %176 = stablehlo.multiply %126#2, %175 : tensor<5xf32>
    %177 = stablehlo.divide %176, %90 : tensor<5xf32>
    %178 = stablehlo.constant dense<4.0> : tensor<1xf32>
    %179 = stablehlo.reshape %178 : (tensor<1xf32>) -> tensor<f32>
    %180 = stablehlo.compare LT, %179, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %181 = stablehlo.constant dense<5.0> : tensor<f32>
    %182 = stablehlo.select %180, %181, %179 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %183 = stablehlo.subtract %182, %8 : tensor<f32>
    %184 = stablehlo.multiply %10, %183 : tensor<f32>
    %185 = stablehlo.sqrt %184 : tensor<f32>
    %186 = stablehlo.divide %1, %185 : tensor<f32>
    %187, %188 = stablehlo.rng_bit_generator %165, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %189 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %190 = stablehlo.shift_right_logical %188, %189 : tensor<128x5xui32>
    %191 = stablehlo.convert %190 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %192 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %193 = stablehlo.multiply %191, %192 : tensor<128x5xf32>
    %194 = stablehlo.multiply %193, %21 : tensor<128x5xf32>
    %195 = stablehlo.subtract %194, %22 : tensor<128x5xf32>
    %196 = chlo.erf_inv %195 : tensor<128x5xf32> -> tensor<128x5xf32>
    %197 = stablehlo.constant dense<1.4142135> : tensor<128x5xf32>
    %198 = stablehlo.multiply %196, %197 : tensor<128x5xf32>
    %199, %200 = stablehlo.rng_bit_generator %187, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x5xui32>)
    %201 = stablehlo.constant dense<9> : tensor<128x5xui32>
    %202 = stablehlo.shift_right_logical %200, %201 : tensor<128x5xui32>
    %203 = stablehlo.convert %202 : (tensor<128x5xui32>) -> tensor<128x5xf32>
    %204 = stablehlo.constant dense<1.1920929E-7> : tensor<128x5xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<128x5xf32>
    %206 = stablehlo.constant dense<false> : tensor<5xi1>
    %210:3 = stablehlo.while(%207 = %35, %208 = %206, %209 = %37) : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    cond {
      %211 = stablehlo.constant dense<128> : tensor<i32>
      %212 = stablehlo.compare LT, %207, %211, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %213 = stablehlo.constant dense<true> : tensor<i1>
      %214 = stablehlo.reduce(%208 init: %213) applies stablehlo.and across dimensions = [0] : (tensor<5xi1>, tensor<i1>) -> tensor<i1>
      %215 = stablehlo.not %214 : tensor<i1>
      %216 = stablehlo.and %212, %215 : tensor<i1>
      stablehlo.return %216 : tensor<i1>
    } do {
      %217 = stablehlo.constant dense<0> : tensor<i32>
      %218 = stablehlo.dynamic_slice %198, %207, %217, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %219 = stablehlo.reshape %218 : (tensor<1x5xf32>) -> tensor<5xf32>
      %220 = stablehlo.dynamic_slice %205, %207, %217, sizes = [1, 5] : (tensor<128x5xf32>, tensor<i32>, tensor<i32>) -> tensor<1x5xf32>
      %221 = stablehlo.reshape %220 : (tensor<1x5xf32>) -> tensor<5xf32>
      %222 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %223 = stablehlo.multiply %222, %219 : tensor<5xf32>
      %224 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %225 = stablehlo.add %224, %223 : tensor<5xf32>
      %226 = stablehlo.multiply %225, %225 : tensor<5xf32>
      %227 = stablehlo.multiply %226, %225 : tensor<5xf32>
      %228 = stablehlo.broadcast_in_dim %183, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %229 = stablehlo.multiply %228, %227 : tensor<5xf32>
      %230 = stablehlo.constant dense<0.5> : tensor<f32>
      %231 = stablehlo.multiply %219, %219 : tensor<5xf32>
      %232 = stablehlo.broadcast_in_dim %230, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %233 = stablehlo.multiply %232, %231 : tensor<5xf32>
      %234 = stablehlo.negate %229 : tensor<5xf32>
      %235 = stablehlo.log %227 : tensor<5xf32>
      %236 = stablehlo.multiply %228, %235 : tensor<5xf32>
      %237 = stablehlo.add %233, %228 : tensor<5xf32>
      %238 = stablehlo.add %237, %234 : tensor<5xf32>
      %239 = stablehlo.add %238, %236 : tensor<5xf32>
      %240 = stablehlo.log %221 : tensor<5xf32>
      %241 = stablehlo.compare LT, %240, %239 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %242 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<5xf32>
      %243 = stablehlo.compare GT, %227, %242 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
      %244 = stablehlo.and %241, %243 : tensor<5xi1>
      %245 = stablehlo.select %208, %209, %229 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
      %246 = stablehlo.or %208, %244 : tensor<5xi1>
      %247 = stablehlo.constant dense<1> : tensor<i32>
      %248 = stablehlo.add %207, %247 : tensor<i32>
      stablehlo.return %248, %246, %245 : tensor<i32>, tensor<5xi1>, tensor<5xf32>
    }
    %249, %250 = stablehlo.rng_bit_generator %199, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %251 = stablehlo.constant dense<9> : tensor<5xui32>
    %252 = stablehlo.shift_right_logical %250, %251 : tensor<5xui32>
    %253 = stablehlo.convert %252 : (tensor<5xui32>) -> tensor<5xf32>
    %254 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %255 = stablehlo.multiply %253, %254 : tensor<5xf32>
    %256 = stablehlo.constant dense<0.25> : tensor<f32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %258 = stablehlo.power %255, %257 : tensor<5xf32>
    %259 = stablehlo.select %180, %258, %90 : (tensor<i1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %260 = stablehlo.multiply %210#2, %259 : tensor<5xf32>
    %261 = stablehlo.divide %260, %90 : tensor<5xf32>
    %262 = stablehlo.reshape %93 : (tensor<5xf32>) -> tensor<1x5xf32>
    %263 = stablehlo.reshape %177 : (tensor<5xf32>) -> tensor<1x5xf32>
    %264 = stablehlo.reshape %261 : (tensor<5xf32>) -> tensor<1x5xf32>
    %265 = stablehlo.concatenate %262, %263, %264, dim = 0 : (tensor<1x5xf32>, tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
    %266 = stablehlo.transpose %265, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %267 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %268 = stablehlo.reduce(%266 init: %267) applies stablehlo.add across dimensions = [1] : (tensor<5x3xf32>, tensor<f32>) -> tensor<5xf32>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %270 = stablehlo.divide %266, %269 : tensor<5x3xf32>
    return %270, %249 : tensor<5x3xf32>, tensor<2xui64>
  }
}
