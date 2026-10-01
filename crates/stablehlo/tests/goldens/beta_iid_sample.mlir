module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %7 = stablehlo.constant dense<"0x5555D53F"> : tensor<f32>
    %11 = stablehlo.constant dense<"0xA532843E"> : tensor<f32>
    %12, %13 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %14 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %15 = stablehlo.shift_right_logical %13, %14 : tensor<128x4xui32>
    %16 = stablehlo.convert %15 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %17 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<128x4xf32>
    %19 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %20 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %21 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %22 = stablehlo.subtract %21, %20 : tensor<128x4xf32>
    %23 = chlo.erf_inv %22 : tensor<128x4xf32> -> tensor<128x4xf32>
    %24 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %25 = stablehlo.multiply %23, %24 : tensor<128x4xf32>
    %26, %27 = stablehlo.rng_bit_generator %12, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %28 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %29 = stablehlo.shift_right_logical %27, %28 : tensor<128x4xui32>
    %30 = stablehlo.convert %29 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %31 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<128x4xf32>
    %33 = stablehlo.constant dense<0> : tensor<i32>
    %34 = stablehlo.constant dense<false> : tensor<4xi1>
    %35 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %39:3 = stablehlo.while(%36 = %33, %37 = %34, %38 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %40 = stablehlo.constant dense<128> : tensor<i32>
      %41 = stablehlo.compare LT, %36, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %42 = stablehlo.constant dense<true> : tensor<i1>
      %43 = stablehlo.reduce(%37 init: %42) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %44 = stablehlo.not %43 : tensor<i1>
      %45 = stablehlo.and %41, %44 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.constant dense<0> : tensor<i32>
      %47 = stablehlo.dynamic_slice %25, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %48 = stablehlo.reshape %47 : (tensor<1x4xf32>) -> tensor<4xf32>
      %49 = stablehlo.dynamic_slice %32, %36, %46, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x4xf32>) -> tensor<4xf32>
      %51 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %52 = stablehlo.multiply %51, %48 : tensor<4xf32>
      %53 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %54 = stablehlo.add %53, %52 : tensor<4xf32>
      %55 = stablehlo.multiply %54, %54 : tensor<4xf32>
      %56 = stablehlo.multiply %55, %54 : tensor<4xf32>
      %57 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %58 = stablehlo.multiply %57, %56 : tensor<4xf32>
      %59 = stablehlo.constant dense<0.5> : tensor<f32>
      %60 = stablehlo.multiply %48, %48 : tensor<4xf32>
      %61 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %62 = stablehlo.multiply %61, %60 : tensor<4xf32>
      %63 = stablehlo.negate %58 : tensor<4xf32>
      %64 = stablehlo.log %56 : tensor<4xf32>
      %65 = stablehlo.multiply %57, %64 : tensor<4xf32>
      %66 = stablehlo.add %62, %57 : tensor<4xf32>
      %67 = stablehlo.add %66, %63 : tensor<4xf32>
      %68 = stablehlo.add %67, %65 : tensor<4xf32>
      %69 = stablehlo.log %50 : tensor<4xf32>
      %70 = stablehlo.compare LT, %69, %68 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %71 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %72 = stablehlo.compare GT, %56, %71 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %73 = stablehlo.and %70, %72 : tensor<4xi1>
      %74 = stablehlo.select %37, %38, %58 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %75 = stablehlo.or %37, %73 : tensor<4xi1>
      %76 = stablehlo.constant dense<1> : tensor<i32>
      %77 = stablehlo.add %36, %76 : tensor<i32>
      stablehlo.return %77, %75, %74 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %78, %79 = stablehlo.rng_bit_generator %26, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %80 = stablehlo.constant dense<9> : tensor<4xui32>
    %81 = stablehlo.shift_right_logical %79, %80 : tensor<4xui32>
    %82 = stablehlo.convert %81 : (tensor<4xui32>) -> tensor<4xf32>
    %83 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %84 = stablehlo.multiply %82, %83 : tensor<4xf32>
    %88 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %89 = stablehlo.multiply %39#2, %88 : tensor<4xf32>
    %90 = stablehlo.divide %89, %88 : tensor<4xf32>
    %92 = stablehlo.constant dense<"0xABAA2A40"> : tensor<f32>
    %95 = stablehlo.constant dense<"0xEB05513E"> : tensor<f32>
    %96, %97 = stablehlo.rng_bit_generator %78, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %98 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %99 = stablehlo.shift_right_logical %97, %98 : tensor<128x4xui32>
    %100 = stablehlo.convert %99 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %101 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %102 = stablehlo.multiply %100, %101 : tensor<128x4xf32>
    %103 = stablehlo.multiply %102, %19 : tensor<128x4xf32>
    %104 = stablehlo.subtract %103, %20 : tensor<128x4xf32>
    %105 = chlo.erf_inv %104 : tensor<128x4xf32> -> tensor<128x4xf32>
    %106 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %107 = stablehlo.multiply %105, %106 : tensor<128x4xf32>
    %108, %109 = stablehlo.rng_bit_generator %96, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %110 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %111 = stablehlo.shift_right_logical %109, %110 : tensor<128x4xui32>
    %112 = stablehlo.convert %111 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %113 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<128x4xf32>
    %115 = stablehlo.constant dense<false> : tensor<4xi1>
    %119:3 = stablehlo.while(%116 = %33, %117 = %115, %118 = %35) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %120 = stablehlo.constant dense<128> : tensor<i32>
      %121 = stablehlo.compare LT, %116, %120, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %122 = stablehlo.constant dense<true> : tensor<i1>
      %123 = stablehlo.reduce(%117 init: %122) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %124 = stablehlo.not %123 : tensor<i1>
      %125 = stablehlo.and %121, %124 : tensor<i1>
      stablehlo.return %125 : tensor<i1>
    } do {
      %126 = stablehlo.constant dense<0> : tensor<i32>
      %127 = stablehlo.dynamic_slice %107, %116, %126, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %128 = stablehlo.reshape %127 : (tensor<1x4xf32>) -> tensor<4xf32>
      %129 = stablehlo.dynamic_slice %114, %116, %126, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %130 = stablehlo.reshape %129 : (tensor<1x4xf32>) -> tensor<4xf32>
      %131 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %132 = stablehlo.multiply %131, %128 : tensor<4xf32>
      %133 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %134 = stablehlo.add %133, %132 : tensor<4xf32>
      %135 = stablehlo.multiply %134, %134 : tensor<4xf32>
      %136 = stablehlo.multiply %135, %134 : tensor<4xf32>
      %137 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %138 = stablehlo.multiply %137, %136 : tensor<4xf32>
      %139 = stablehlo.constant dense<0.5> : tensor<f32>
      %140 = stablehlo.multiply %128, %128 : tensor<4xf32>
      %141 = stablehlo.broadcast_in_dim %139, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %142 = stablehlo.multiply %141, %140 : tensor<4xf32>
      %143 = stablehlo.negate %138 : tensor<4xf32>
      %144 = stablehlo.log %136 : tensor<4xf32>
      %145 = stablehlo.multiply %137, %144 : tensor<4xf32>
      %146 = stablehlo.add %142, %137 : tensor<4xf32>
      %147 = stablehlo.add %146, %143 : tensor<4xf32>
      %148 = stablehlo.add %147, %145 : tensor<4xf32>
      %149 = stablehlo.log %130 : tensor<4xf32>
      %150 = stablehlo.compare LT, %149, %148 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %151 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %152 = stablehlo.compare GT, %136, %151 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %153 = stablehlo.and %150, %152 : tensor<4xi1>
      %154 = stablehlo.select %117, %118, %138 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %155 = stablehlo.or %117, %153 : tensor<4xi1>
      %156 = stablehlo.constant dense<1> : tensor<i32>
      %157 = stablehlo.add %116, %156 : tensor<i32>
      stablehlo.return %157, %155, %154 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %158, %159 = stablehlo.rng_bit_generator %108, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %160 = stablehlo.constant dense<9> : tensor<4xui32>
    %161 = stablehlo.shift_right_logical %159, %160 : tensor<4xui32>
    %162 = stablehlo.convert %161 : (tensor<4xui32>) -> tensor<4xf32>
    %163 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %164 = stablehlo.multiply %162, %163 : tensor<4xf32>
    %168 = stablehlo.multiply %119#2, %88 : tensor<4xf32>
    %169 = stablehlo.divide %168, %88 : tensor<4xf32>
    %170 = stablehlo.add %90, %169 : tensor<4xf32>
    %171 = stablehlo.divide %90, %170 : tensor<4xf32>
    return %171, %158 : tensor<4xf32>, tensor<2xui64>
  }
}
