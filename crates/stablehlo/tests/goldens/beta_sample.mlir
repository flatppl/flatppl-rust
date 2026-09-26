module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<false> : tensor<i1>
    %6 = stablehlo.constant dense<1.6666666269302368> : tensor<f32>
    %10 = stablehlo.constant dense<0.25819888710975647> : tensor<f32>
    %11, %12 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %13 = stablehlo.constant dense<9> : tensor<128xui32>
    %14 = stablehlo.shift_right_logical %12, %13 : tensor<128xui32>
    %15 = stablehlo.convert %14 : (tensor<128xui32>) -> tensor<128xf32>
    %16 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %17 = stablehlo.multiply %15, %16 : tensor<128xf32>
    %18 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %19 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %20 = stablehlo.multiply %17, %18 : tensor<128xf32>
    %21 = stablehlo.subtract %20, %19 : tensor<128xf32>
    %22 = chlo.erf_inv %21 : tensor<128xf32> -> tensor<128xf32>
    %23 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %24 = stablehlo.multiply %22, %23 : tensor<128xf32>
    %25, %26 = stablehlo.rng_bit_generator %11, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %27 = stablehlo.constant dense<9> : tensor<128xui32>
    %28 = stablehlo.shift_right_logical %26, %27 : tensor<128xui32>
    %29 = stablehlo.convert %28 : (tensor<128xui32>) -> tensor<128xf32>
    %30 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<128xf32>
    %32 = stablehlo.constant dense<0> : tensor<i32>
    %36:3 = stablehlo.while(%33 = %32, %34 = %4, %35 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %37 = stablehlo.constant dense<128> : tensor<i32>
      %38 = stablehlo.compare LT, %33, %37, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %39 = stablehlo.not %34 : tensor<i1>
      %40 = stablehlo.and %39, %38 : tensor<i1>
      stablehlo.return %40 : tensor<i1>
    } do {
      %41 = stablehlo.dynamic_slice %24, %33, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %42 = stablehlo.reshape %41 : (tensor<1xf32>) -> tensor<f32>
      %43 = stablehlo.dynamic_slice %31, %33, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %44 = stablehlo.reshape %43 : (tensor<1xf32>) -> tensor<f32>
      %45 = stablehlo.multiply %10, %42 : tensor<f32>
      %46 = stablehlo.add %2, %45 : tensor<f32>
      %47 = stablehlo.multiply %46, %46 : tensor<f32>
      %48 = stablehlo.multiply %47, %46 : tensor<f32>
      %49 = stablehlo.multiply %6, %48 : tensor<f32>
      %50 = stablehlo.constant dense<0.5> : tensor<f32>
      %51 = stablehlo.multiply %42, %42 : tensor<f32>
      %52 = stablehlo.multiply %50, %51 : tensor<f32>
      %53 = stablehlo.negate %49 : tensor<f32>
      %54 = stablehlo.log %48 : tensor<f32>
      %55 = stablehlo.multiply %6, %54 : tensor<f32>
      %56 = stablehlo.add %52, %6 : tensor<f32>
      %57 = stablehlo.add %56, %53 : tensor<f32>
      %58 = stablehlo.add %57, %55 : tensor<f32>
      %59 = stablehlo.log %44 : tensor<f32>
      %60 = stablehlo.compare LT, %59, %58 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %61 = stablehlo.compare GT, %48, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %62 = stablehlo.and %60, %61 : tensor<i1>
      %63 = stablehlo.constant dense<1> : tensor<i32>
      %64 = stablehlo.add %33, %63 : tensor<i32>
      stablehlo.return %64, %62, %49 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %65, %66 = stablehlo.rng_bit_generator %25, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %67 = stablehlo.constant dense<9> : tensor<ui32>
    %68 = stablehlo.shift_right_logical %66, %67 : tensor<ui32>
    %69 = stablehlo.convert %68 : (tensor<ui32>) -> tensor<f32>
    %70 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %71 = stablehlo.multiply %69, %70 : tensor<f32>
    %74 = stablehlo.multiply %36#2, %2 : tensor<f32>
    %75 = stablehlo.divide %74, %2 : tensor<f32>
    %77 = stablehlo.constant dense<2.6666667461395264> : tensor<f32>
    %80 = stablehlo.constant dense<0.20412413775920868> : tensor<f32>
    %81, %82 = stablehlo.rng_bit_generator %65, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %83 = stablehlo.constant dense<9> : tensor<128xui32>
    %84 = stablehlo.shift_right_logical %82, %83 : tensor<128xui32>
    %85 = stablehlo.convert %84 : (tensor<128xui32>) -> tensor<128xf32>
    %86 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %87 = stablehlo.multiply %85, %86 : tensor<128xf32>
    %88 = stablehlo.multiply %87, %18 : tensor<128xf32>
    %89 = stablehlo.subtract %88, %19 : tensor<128xf32>
    %90 = chlo.erf_inv %89 : tensor<128xf32> -> tensor<128xf32>
    %91 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %92 = stablehlo.multiply %90, %91 : tensor<128xf32>
    %93, %94 = stablehlo.rng_bit_generator %81, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %95 = stablehlo.constant dense<9> : tensor<128xui32>
    %96 = stablehlo.shift_right_logical %94, %95 : tensor<128xui32>
    %97 = stablehlo.convert %96 : (tensor<128xui32>) -> tensor<128xf32>
    %98 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %99 = stablehlo.multiply %97, %98 : tensor<128xf32>
    %103:3 = stablehlo.while(%100 = %32, %101 = %4, %102 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %104 = stablehlo.constant dense<128> : tensor<i32>
      %105 = stablehlo.compare LT, %100, %104, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %106 = stablehlo.not %101 : tensor<i1>
      %107 = stablehlo.and %106, %105 : tensor<i1>
      stablehlo.return %107 : tensor<i1>
    } do {
      %108 = stablehlo.dynamic_slice %92, %100, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %109 = stablehlo.reshape %108 : (tensor<1xf32>) -> tensor<f32>
      %110 = stablehlo.dynamic_slice %99, %100, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %111 = stablehlo.reshape %110 : (tensor<1xf32>) -> tensor<f32>
      %112 = stablehlo.multiply %80, %109 : tensor<f32>
      %113 = stablehlo.add %2, %112 : tensor<f32>
      %114 = stablehlo.multiply %113, %113 : tensor<f32>
      %115 = stablehlo.multiply %114, %113 : tensor<f32>
      %116 = stablehlo.multiply %77, %115 : tensor<f32>
      %117 = stablehlo.constant dense<0.5> : tensor<f32>
      %118 = stablehlo.multiply %109, %109 : tensor<f32>
      %119 = stablehlo.multiply %117, %118 : tensor<f32>
      %120 = stablehlo.negate %116 : tensor<f32>
      %121 = stablehlo.log %115 : tensor<f32>
      %122 = stablehlo.multiply %77, %121 : tensor<f32>
      %123 = stablehlo.add %119, %77 : tensor<f32>
      %124 = stablehlo.add %123, %120 : tensor<f32>
      %125 = stablehlo.add %124, %122 : tensor<f32>
      %126 = stablehlo.log %111 : tensor<f32>
      %127 = stablehlo.compare LT, %126, %125 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %128 = stablehlo.compare GT, %115, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %129 = stablehlo.and %127, %128 : tensor<i1>
      %130 = stablehlo.constant dense<1> : tensor<i32>
      %131 = stablehlo.add %100, %130 : tensor<i32>
      stablehlo.return %131, %129, %116 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %132, %133 = stablehlo.rng_bit_generator %93, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %134 = stablehlo.constant dense<9> : tensor<ui32>
    %135 = stablehlo.shift_right_logical %133, %134 : tensor<ui32>
    %136 = stablehlo.convert %135 : (tensor<ui32>) -> tensor<f32>
    %137 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %138 = stablehlo.multiply %136, %137 : tensor<f32>
    %141 = stablehlo.multiply %103#2, %2 : tensor<f32>
    %142 = stablehlo.divide %141, %2 : tensor<f32>
    %143 = stablehlo.add %75, %142 : tensor<f32>
    %144 = stablehlo.divide %75, %143 : tensor<f32>
    return %144, %132 : tensor<f32>, tensor<2xui64>
  }
}
